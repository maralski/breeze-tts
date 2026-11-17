// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Where each segment of each branch lands: rows of the backbone input, rows of the text
/// encoder's input, frames of the uploaded reference codes.
struct PromptLayout {
    struct Text {
        let ids: [Int32]
        /// First backbone row.
        let row: Int
        /// First row in the text encoder's input buffer; nil when an identical segment earlier in
        /// the prompt is encoded already, whose projected rows `copyOf` names.
        let encoderRow: Int?
        let copyOf: (branch: Int, row: Int)?
    }
    struct Audio {
        /// First frame in the uploaded codes; the end-of-audio row follows the last frame.
        let frame: Int
        let frames: Int
        let row: Int
    }
    struct Branch {
        let rows: Int
        let texts: [Text]
        let audios: [Audio]
    }

    let branches: [Branch]
    /// Rows the text encoder runs on in all.
    let encoderRows: Int
    /// uint32 [frames + 1, 16]: every reference frame, then the all-zero frame the end-of-audio
    /// rows embed.
    let codes: [UInt32]
    var longestSegment: Int { branches.flatMap(\.texts).map(\.ids.count).max() ?? 0 }
    var longestBranch: Int { branches.map(\.rows).max()! }

    init(_ prompt: BreezePrompt) {
        var encoderRows = 0
        var codes: [UInt32] = []
        var seen: [[Int32]: (branch: Int, row: Int)] = [:]
        var branches: [Branch] = []
        for (b, branch) in ([prompt.positive] + (prompt.guidance.map { [$0.negative] } ?? [])).enumerated() {
            var row = 0
            var texts: [Text] = [], audios: [Audio] = []
            for segment in branch.segments {
                switch segment {
                case .text(let ids):
                    if let earlier = seen[ids] {
                        texts.append(Text(ids: ids, row: row, encoderRow: nil, copyOf: earlier))
                    } else {
                        texts.append(Text(ids: ids, row: row, encoderRow: encoderRows, copyOf: nil))
                        seen[ids] = (b, row)
                        encoderRows += ids.count
                    }
                    row += ids.count
                case .audio(let frames):
                    audios.append(Audio(frame: codes.count / BreezeShape.codebooks, frames: frames.count, row: row))
                    for frame in frames { codes += frame.map { UInt32(UInt16(bitPattern: $0)) } }
                    row += frames.count + 1
                }
            }
            precondition(row == branch.count)
            branches.append(Branch(rows: row, texts: texts, audios: audios))
        }
        self.branches = branches
        self.encoderRows = encoderRows
        self.codes = codes + [UInt32](repeating: 0, count: BreezeShape.codebooks)
    }
}

/// How the frames are drawn. Guidance is not here: `BreezePrompt.guidance` is its one home.
struct SynthesisOptions {
    var maxFrames = 1500
    var temperature: Float = 0.9
    var topK = 50
    var repetitionPenalty: Float = 1.1
    var greedy = false
    var uniforms = UniformSource.seeded(42)

    func frameSettings(guidance: Float?) -> FrameSettings {
        let s = SamplerSettings(temperature: temperature, topK: topK, repetitionPenalty: repetitionPenalty, greedy: greedy)
        return FrameSettings(guidance: guidance, backbone: s, depth: s)
    }
}

/// The uniform each draw consumes. Frame k's 16 values drive codebooks 0 ... 15 in order.
enum UniformSource {
    case seeded(UInt64)
    /// Per frame, 16 values in [0, 1).
    case recorded([[Float]])

    func generator() -> (Int) -> [Float] {
        switch self {
        case .seeded(let seed):
            var rng = SplitMix64(seed: seed)
            return { _ in (0..<BreezeShape.codebooks).map { _ in rng.unit() } }
        case .recorded(let frames):
            return { frames[$0] }
        }
    }
}

struct SplitMix64 {
    private var state: UInt64
    /// Seeded as MingTTS seeds its noise stream.
    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// The top 24 bits over 2^24: exactly representable, below 1 in fp32.
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
}

struct SynthesisReport {
    let frames: Int
    let stoppedOnEndOfSpeech: Bool
    /// Text encoding, prompt assembly and backbone prefill, one command buffer, wall clock.
    let prefillSeconds: Double
    /// Per emitted frame, wall clock from the previous frame (or the prefill) to its codes.
    let frameSeconds: [Double]
}

/// Text in, codes out: the text encoder, prompt assembly, backbone prefill and the frame loop,
/// over one loaded checkpoint. Not thread-safe; one request at a time.
final class Synthesizer {
    let ctx: MetalContext
    let kernels: Kernels
    let frameWeights: FrameWeights
    let text: TextEncoder
    let embedding: TextEmbedding
    /// fp32 [2 * 2048 + 8, 1152]: the text encoder's input and residual rows, every segment of
    /// the request at its own rows.
    let encoderInput: MTLBuffer
    /// uint32 reference codes of the request and the zero frame.
    let codes: MTLBuffer
    /// fp32 [2048 + 8, 2048] per branch: the backbone's input rows, then its residual stream.
    let inputs: [MTLBuffer]
    let prefillWork: DecoderWorkspace
    private var loops: [Int: (loop: FrameLoop, prefill: DecoderForward)] = [:]

    static let positions = BreezePrompt.backbonePositions

    init(directory: URL, ctx: MetalContext? = nil) throws {
        let ctx = try ctx ?? MetalContext()
        self.ctx = ctx
        kernels = try Kernels(ctx: ctx)
        let store = try ShardedCheckpoint(directory: directory)
        frameWeights = try FrameWeights(store, ctx: ctx)
        let config = TextEncoderConfig.t5gemma2
        let textWeights = try WeightArena.load(store, ctx: ctx, label: "weights.text") { try TextEncoderWeights($0, config: config) }
        text = try TextEncoder(ctx: ctx, weights: textWeights, config: config, maxRows: Self.positions)
        embedding = try TextEmbedding(store, config: config)
        encoderInput = try ctx.makeBuffer(bytes: (2 * Self.positions + 8) * config.hidden * 4, label: "prompt.encoderInput")
        codes = try ctx.makeBuffer(bytes: (2 * Self.positions + 1) * BreezeShape.codebooks * 4, label: "prompt.codes")
        inputs = try (0..<2).map { try ctx.makeBuffer(bytes: (Self.positions + 8) * BreezeShape.embedWidth * 4, label: "prompt.inputs\($0)") }
        prefillWork = try DecoderWorkspace(ctx: ctx, config: .backbone, maxRows: Self.positions, label: "prefill")
    }

    /// The frame loop for one or two sequences, and a prefill forward over its backbone cache.
    func loop(sequences: Int) throws -> (loop: FrameLoop, prefill: DecoderForward) {
        if let l = loops[sequences] { return l }
        let loop = try FrameLoop(ctx: ctx, weights: frameWeights, sequences: sequences, maxFrames: Self.positions)
        let prefill = try DecoderForward(ctx: ctx, weights: frameWeights.backbone, cache: loop.backbone.cache, work: prefillWork)
        loops[sequences] = (loop, prefill)
        return (loop, prefill)
    }

    func inputRows(_ branch: Int) -> ActView { ActView(inputs[branch], ld: BreezeShape.embedWidth) }

    /// CPU half of the prompt: the text embeddings and the reference codes into their buffers.
    func upload(_ layout: PromptLayout) throws {
        precondition(layout.encoderRows <= 2 * Self.positions && layout.codes.count * 4 <= codes.length)
        let x = encoderInput.contents().bindMemory(to: Float.self, capacity: encoderInput.length / 4)
        for t in layout.branches.flatMap(\.texts) {
            if let r = t.encoderRow { try embedding.gather(t.ids, into: x + r * text.config.hidden) }
        }
        _ = layout.codes.withUnsafeBytes { memcpy(codes.contents(), $0.baseAddress!, $0.count) }
    }

    /// Every branch's backbone input rows: projected text-encoder rows, summed code embeddings,
    /// the end-of-audio rows. `afterLayer(segment, layer)` and `afterNorm` reach into the text
    /// encoder of each encoded segment, numbered in layout order.
    func encodePrompt(_ e: Encoder, _ layout: PromptLayout, afterLayer: ((Int, Int) -> Void)? = nil, afterNorm: ((Int, ActView) -> Void)? = nil) {
        let w = BreezeShape.embedWidth, hidden = text.config.hidden
        var segment = 0
        for (b, branch) in layout.branches.enumerated() {
            let rows = inputRows(b)
            for t in branch.texts {
                if let r = t.encoderRow {
                    let s = segment
                    text.encode(e, x: ActView(encoderInput, offset: r * hidden * 4, ld: hidden), rows: t.ids.count, out: rows.row(t.row),
                                afterLayer: afterLayer.map { f in { f(s, $0) } }, afterNorm: afterNorm.map { f in { f(s, $0) } })
                    segment += 1
                } else if let source = t.copyOf {
                    e.copy(inputRows(source.branch).row(source.row), count: t.ids.count * w, to: rows.row(t.row))
                }
            }
            for a in branch.audios {
                e.embedCodes(table: frameWeights.embedding, width: w, vocab: BreezeShape.codebookVocab, codes: codes, codesOffset: a.frame * BreezeShape.codebooks * 4,
                             firstCodebook: 0, count: BreezeShape.codebooks, rows: a.frames, rowStride: BreezeShape.codebooks, rows.row(a.row))
                e.embedCodes(table: frameWeights.embedding, width: w, vocab: BreezeShape.codebookVocab, codes: codes, codesOffset: (layout.codes.count - BreezeShape.codebooks) * 4,
                             firstCodebook: 0, count: BreezeShape.codebooks, rows: 1, rows.row(a.row + a.frames))
            }
        }
    }

    /// Each branch through the backbone alone at positions 0 ..< rows, into its own sequence of
    /// the loop's cache; the final-norm hidden state of its last row lands in `depthInput`,
    /// where frame 0 reads it. Resets the cache first.
    func encodePrefill(_ e: Encoder, _ layout: PromptLayout, loop: FrameLoop, prefill: DecoderForward, afterLayer: ((Int, Int) -> Void)? = nil) {
        let w = BreezeShape.embedWidth
        loop.backbone.cache.reset()
        for (b, branch) in layout.branches.enumerated() {
            prefill.encode(e, x: inputRows(b), rows: branch.rows, sequences: b..<(b + 1), out: ActView(loop.depthInput, offset: b * 2 * w * 4, ld: 2 * w),
                           afterLayer: afterLayer.map { f in { f(b, $0) } })
        }
    }

    func commit(_ body: (Encoder) throws -> Void) throws -> MTLCommandBuffer {
        guard let cb = ctx.queue.makeCommandBuffer(), let ce = cb.makeComputeCommandEncoder(dispatchType: .serial) else {
            throw BreezeTTSError.gpuFailure("cannot create a command buffer")
        }
        try body(Encoder(k: kernels, ce: ce))
        ce.endEncoding()
        cb.commit()
        return cb
    }

    static func wait(_ cb: MTLCommandBuffer) throws {
        cb.waitUntilCompleted()
        if let error = cb.error { throw BreezeTTSError.gpuFailure("\(error)") }
    }

    /// Generates frames for `prompt` until end of speech, `options.maxFrames`, or the backbone's
    /// position limit, handing each frame's 16 codes to `onFrame` as soon as it completes. Two
    /// frames are in flight at a time; a frame encoded past the stop is discarded.
    func run(_ prompt: BreezePrompt, options: SynthesisOptions, onFrame: ([Int]) throws -> Void) throws -> SynthesisReport {
        let layout = PromptLayout(prompt)
        let (loop, prefill) = try loop(sequences: layout.branches.count)
        let start = Date()
        try upload(layout)
        try Self.wait(try commit { e in
            encodePrompt(e, layout)
            encodePrefill(e, layout, loop: loop, prefill: prefill)
        })
        var last = Date()
        let prefillSeconds = last.timeIntervalSince(start)

        loop.begin(positions: layout.branches.map(\.rows))
        let settings = options.frameSettings(guidance: prompt.guidance?.scale)
        let uniforms = options.uniforms.generator()
        // Frame k's backbone step writes position rows + k - 1; the reference stops before 2047.
        let limit = min(options.maxFrames, Self.positions - 1 - layout.longestBranch)
        var inFlight: [MTLCommandBuffer] = []
        var encoded = 0, done = 0
        var stopped = false
        var frameSeconds: [Double] = []
        defer { for cb in inFlight { cb.waitUntilCompleted() } }
        while done < limit {
            while encoded < limit && inFlight.count < 2 {
                let f = encoded
                loop.setUniforms(frame: f, uniforms(f))
                inFlight.append(try commit { e in
                    if f == 0 { loop.encodeCodes(e, frame: 0, settings: settings) } else { loop.encodeFrame(e, frame: f, settings: settings) }
                })
                encoded += 1
            }
            try Self.wait(inFlight.removeFirst())
            guard loop.statusWord == 0 else { throw BreezeTTSError.gpuFailure("frame \(done) met NaN or empty logits") }
            let frame = loop.frameCodes(done)
            if frame[0] == BreezeShape.endOfSpeech {
                stopped = true
                break
            }
            let now = Date()
            frameSeconds.append(now.timeIntervalSince(last))
            last = now
            try onFrame(frame)
            done += 1
        }
        return SynthesisReport(frames: done, stoppedOnEndOfSpeech: stopped, prefillSeconds: prefillSeconds, frameSeconds: frameSeconds)
    }
}
