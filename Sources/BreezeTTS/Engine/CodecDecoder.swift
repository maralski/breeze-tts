// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// The Qwen3-TTS 12 Hz tokenizer's decoder as a streaming GPU program: 16 codes per frame in,
/// 1920 samples of 24 kHz mono PCM out.
///
///     codes -> split RVQ lookup -> pre_conv -> 8-layer transformer (72-frame window)
///           -> two x2 upsample stages (transposed conv + ConvNeXt)
///           -> vocoder: conv, four blocks (SnakeBeta, transposed conv x8/x5/x4/x3, three
///              dilated residual units), SnakeBeta, conv -> clamp
///
/// A step decodes one or more consecutive frames. Everything a later step needs stays on the
/// GPU: each causal conv's left context and each transposed conv's previous input row sit in a
/// `CodecRing`, and the transformer's keys and values in ring caches indexed by absolute frame
/// position. Results equal one full forward over the whole stream whatever the step sizes.
///
/// `append` returns dispatches, not a command buffer, so the caller can put a frame's decode in
/// the same command buffer as the token loop that produced its codes.
final class CodecDecoder {
    let ctx: MetalContext
    let kernels: CodecKernels
    let weights: CodecDecoderWeights
    /// The most frames one step decodes.
    let chunkFrames: Int
    /// Frames one stream may hold (the RoPE table's length).
    let maxFrames: Int
    let pcmSlots: Int
    /// fp32 [pcmSlots][chunkFrames * 1920]. Step s since the last reset writes slot s % pcmSlots.
    let pcm: MTLBuffer

    /// Frames whose dispatches have been appended since the last reset.
    private(set) var frames = 0
    private var steps = 0

    private let rope: RopeTable
    private let kvRing: Int
    private let keys: [MTLBuffer]
    private let values: [MTLBuffer]
    private let sums, quantized, preOut, x, xn, qkv, attn, gateUp, mlp, latent, conv0Out: MTLBuffer
    private let preRing, conv0Ring, finalRing: CodecRing

    private struct UpsampleState {
        let out: MTLBuffer
        let dwRing: CodecRing
        let normed: MTLBuffer
        let wide: MTLBuffer
    }

    private struct BlockState {
        /// Time rows per frame entering and leaving the block.
        let rowsIn, rowsOut: Int
        let transRing: CodecRing
        let stream: MTLBuffer
        let branch: MTLBuffer
        let unitRings: [CodecRing]
    }

    private let upsample: [UpsampleState]
    private let blocks: [BlockState]
    private var batchCodes: MTLBuffer?
    private var batchKernels: Kernels?

    /// A stage output of the last appended step: valid once its command buffer has completed
    /// and until the next step runs.
    struct StageView {
        let buffer: MTLBuffer
        let offset: Int
        let rows: Int
        let width: Int
        func floats() -> [Float] { buffer.floats(count: rows * width, offset: offset) }
    }
    private(set) var lastStages: [String: StageView] = [:]

    convenience init(directory: URL, ctx: MetalContext, chunkFrames: Int = 1, maxFrames: Int = 2048, pcmSlots: Int = 8) throws {
        try self.init(ctx: ctx, kernels: try CodecKernels(ctx: ctx), weights: try CodecDecoderWeights(try CodecLoader(directory: directory, ctx: ctx)),
                      chunkFrames: chunkFrames, maxFrames: maxFrames, pcmSlots: pcmSlots)
    }

    init(ctx: MetalContext, kernels: CodecKernels, weights: CodecDecoderWeights, chunkFrames: Int = 1, maxFrames: Int = 2048, pcmSlots: Int = 8) throws {
        precondition(chunkFrames >= 1 && pcmSlots >= 1)
        self.ctx = ctx
        self.kernels = kernels
        self.weights = weights
        self.chunkFrames = chunkFrames
        self.maxFrames = maxFrames
        self.pcmSlots = pcmSlots
        let f = chunkFrames, h = CodecDecoderWeights.hidden, lat = CodecDecoderWeights.latent
        func plain(_ rows: Int, _ width: Int, _ label: String) throws -> MTLBuffer {
            try ctx.makeBuffer(bytes: (rows + CodecRing.slackRows) * width * 4, label: "codec." + label)
        }
        pcm = try ctx.makeBuffer(bytes: pcmSlots * f * CodecShape.samplesPerFrame * 4, label: "codec.pcm")
        rope = try RopeTable(ctx: ctx, positions: maxFrames)
        kvRing = CodecShape.decoderWindow + f - 1
        let kvBytes = CodecDecoderWeights.heads * kvRing * 64 * 4
        keys = try weights.layers.indices.map { try ctx.makeBuffer(bytes: kvBytes, label: "codec.k\($0)") }
        values = try weights.layers.indices.map { try ctx.makeBuffer(bytes: kvBytes, label: "codec.v\($0)") }
        sums = try plain(f, 2 * CodecShape.codebookDim, "sums")
        quantized = try plain(f, h, "quantized")
        preRing = try CodecRing(ctx: ctx, width: h, pad: 2, maxNewRows: f, label: "codec.preRing")
        preOut = try plain(f, lat, "preOut")
        x = try plain(f, h, "x")
        xn = try plain(f, h, "xn")
        qkv = try plain(f, 3 * lat, "qkv")
        attn = try plain(f, lat, "attn")
        gateUp = try plain(f, 2 * lat, "gateUp")
        mlp = try plain(f, lat, "mlp")
        latent = try plain(f, lat, "latent")
        upsample = try (0..<2).map { i in
            let rows = f * (2 << i)
            return UpsampleState(out: try plain(rows, lat, "up\(i)"),
                                 dwRing: try CodecRing(ctx: ctx, width: lat, pad: 6, maxNewRows: rows, label: "codec.up\(i).dw"),
                                 normed: try plain(rows, lat, "up\(i).normed"),
                                 wide: try plain(rows, 4 * lat, "up\(i).wide"))
        }
        conv0Ring = try CodecRing(ctx: ctx, width: lat, pad: 6, maxNewRows: 4 * f, label: "codec.conv0Ring")
        conv0Out = try plain(4 * f, CodecDecoderWeights.vocoderWidth, "conv0")
        var rowsPerFrame = 4, width = CodecDecoderWeights.vocoderWidth
        var states: [BlockState] = []
        for (i, b) in weights.blocks.enumerated() {
            let out = rowsPerFrame * b.stride
            states.append(BlockState(
                rowsIn: rowsPerFrame, rowsOut: out,
                transRing: try CodecRing(ctx: ctx, width: width, pad: 1, maxNewRows: rowsPerFrame * f, label: "codec.block\(i).in"),
                stream: try plain(out * f, b.channels, "block\(i).stream"),
                branch: try plain(out * f, b.channels, "block\(i).branch"),
                unitRings: try b.units.enumerated().map { u, unit in
                    try CodecRing(ctx: ctx, width: b.channels, pad: 6 * unit.dilation, maxNewRows: out * f, label: "codec.block\(i).unit\(u)")
                }))
            rowsPerFrame = out
            width = b.channels
        }
        blocks = states
        finalRing = try CodecRing(ctx: ctx, width: width, pad: 6, maxNewRows: rowsPerFrame * f, label: "codec.finalRing")
        precondition(rowsPerFrame == CodecShape.samplesPerFrame)
    }

    private var rings: [CodecRing] {
        [preRing] + upsample.map(\.dwRing) + [conv0Ring] + blocks.flatMap { [$0.transRing] + $0.unitRings } + [finalRing]
    }

    /// Starts a new stream. The first step appended afterwards zeroes the conv histories on the
    /// GPU, so this is safe while earlier command buffers are still running.
    func reset() {
        frames = 0
        steps = 0
        lastStages = [:]
    }

    /// The dispatches that decode the next `count` frames of the stream. `codes` holds uint32
    /// [count][16] at `codesOffset` bytes, written by the time the dispatches run. The samples
    /// land in `pcm` at `pcmOffset` bytes once the command buffer completes.
    func append(codes: MTLBuffer, codesOffset: Int, count f: Int = 1) -> (dispatches: [(Encoder) -> Void], pcmOffset: Int, samples: Int) {
        precondition(f >= 1 && f <= chunkFrames, "a step decodes 1...\(chunkFrames) frames")
        precondition(frames + f <= maxFrames, "the stream is limited to \(maxFrames) frames")
        let k = kernels, w = weights, n0 = frames
        let h = CodecDecoderWeights.hidden, lat = CodecDecoderWeights.latent, heads = CodecDecoderWeights.heads
        var list: [(Encoder) -> Void] = []
        var stages: [String: StageView] = [:]
        if n0 == 0 {
            for ring in rings { list.append { k.zero($0, ring.buffer, bytes: ring.storedBytes) } }
        }

        let sumsView = ActView(sums, ld: 2 * CodecShape.codebookDim), quantizedView = ActView(quantized, ld: h)
        list.append { k.rvqSum($0, codebooks: w.codebooks, codes: codes, codesOffset: codesOffset, rows: f, sumsView) }
        let preRing = self.preRing
        list.append { k.product($0, CodecWindow(sumsView), rows: f, w.quantizerProj, main: quantizedView, act: preRing.output(head: n0, .identity)) }
        stages["quantizer"] = StageView(buffer: quantized, offset: 0, rows: f, width: h)
        let preView = ActView(preOut, ld: lat)
        list.append { k.product($0, preRing.window(head: n0), rows: f, w.preConv, main: preView) }
        stages["pre_conv"] = StageView(buffer: preOut, offset: 0, rows: f, width: lat)

        let xView = ActView(x, ld: h), xnView = ActView(xn, ld: h), qkvView = ActView(qkv, ld: 3 * lat)
        let attnView = ActView(attn, ld: lat), gateUpView = ActView(gateUp, ld: 2 * lat), mlpView = ActView(mlp, ld: lat)
        list.append { k.product($0, CodecWindow(preView), rows: f, w.inputProj, main: xView) }
        let rope = self.rope, kvRing = self.kvRing, eps = CodecDecoderWeights.rmsEps
        for (i, layer) in w.layers.enumerated() {
            let keys = self.keys[i], values = self.values[i]
            list.append { k.norm($0, xView, rows: f, width: h, weight: layer.norm1, xnView, eps: eps) }
            list.append { k.product($0, CodecWindow(xnView), rows: f, layer.qkv, main: qkvView) }
            list.append { k.ropeKV($0, qkvView, rows: f, heads: heads, pos0: n0, ring: kvRing, k: keys, v: values, table: rope) }
            list.append { k.attention($0, qkvView, rows: f, heads: heads, pos0: n0, window: CodecShape.decoderWindow, ring: kvRing, k: keys, v: values, attnView) }
            list.append { k.product($0, CodecWindow(attnView), rows: f, layer.o, main: xView, accumulate: true) }
            list.append { k.norm($0, xView, rows: f, width: h, weight: layer.norm2, xnView, eps: eps) }
            list.append { k.product($0, CodecWindow(xnView), rows: f, layer.gateUp, main: gateUpView) }
            list.append { k.swiglu($0, gateUpView, rows: f, hidden: lat, mlpView) }
            list.append { k.product($0, CodecWindow(mlpView), rows: f, layer.down, main: xView, accumulate: true) }
        }
        let latentView = ActView(latent, ld: lat)
        list.append { k.norm($0, xView, rows: f, width: h, weight: w.finalNorm, xnView, eps: eps) }
        list.append { k.product($0, CodecWindow(xnView), rows: f, w.outputProj, main: latentView) }
        stages["pre_transformer"] = StageView(buffer: latent, offset: 0, rows: f, width: lat)

        let conv0Ring = self.conv0Ring
        var input = latentView
        for (i, (uw, us)) in zip(w.upsample, upsample).enumerated() {
            let rowsIn = f << i, rows = 2 * rowsIn, head = n0 * (2 << i)
            let source = input, outView = ActView(us.out, ld: lat), normedView = ActView(us.normed, ld: lat)
            list.append { k.product($0, CodecWindow(source), rows: rowsIn, uw.trans, main: outView.strided(2 * lat), act: us.dwRing.output(head: head, .identity)) }
            list.append {
                k.dwconvLN($0, us.dwRing.window(head: head), rows: rows, channels: lat, taps: 7, w: uw.dw, bias: uw.dwBias, lnWeight: uw.lnWeight, lnBias: uw.lnBias,
                           normedView, eps: CodecDecoderWeights.layerNormEps)
            }
            list.append { k.product($0, CodecWindow(normedView), rows: rows, uw.pw1, act: CodecActOutput(buffer: us.wide, ld: 4 * lat, channels: 4 * lat, activation: .gelu)) }
            let last = i == w.upsample.count - 1
            list.append {
                k.product($0, CodecWindow(us.wide, ld: 4 * lat), rows: rows, uw.pw2, main: outView, accumulate: true, act: last ? conv0Ring.output(head: head, .identity) : nil)
            }
            stages["upsample\(i)"] = StageView(buffer: us.out, offset: 0, rows: rows, width: lat)
            input = outView
        }

        let conv0View = ActView(conv0Out, ld: CodecDecoderWeights.vocoderWidth), firstBlock = blocks[0], finalRing = self.finalRing
        list.append {
            k.product($0, conv0Ring.window(head: 4 * n0), rows: 4 * f, w.conv0, main: conv0View, act: firstBlock.transRing.output(head: 4 * n0, w.blocks[0].inAct))
        }
        stages["decoder0"] = StageView(buffer: conv0Out, offset: 0, rows: 4 * f, width: CodecDecoderWeights.vocoderWidth)
        for (i, (bw, bs)) in zip(w.blocks, blocks).enumerated() {
            let c = bw.channels, headIn = n0 * bs.rowsIn, head = n0 * bs.rowsOut, rows = bs.rowsOut * f
            let stream = ActView(bs.stream, ld: c)
            list.append {
                k.product($0, bs.transRing.window(head: headIn), rows: bs.rowsIn * f, bw.trans, main: stream.strided(bw.stride * c), act: bs.unitRings[0].output(head: head, bw.units[0].act1))
            }
            for (u, unit) in bw.units.enumerated() {
                let next: CodecActOutput
                if u + 1 < bw.units.count {
                    next = bs.unitRings[u + 1].output(head: head, bw.units[u + 1].act1)
                } else if i + 1 < blocks.count {
                    next = blocks[i + 1].transRing.output(head: head, w.blocks[i + 1].inAct)
                } else {
                    next = finalRing.output(head: head, w.finalAct)
                }
                list.append {
                    k.product($0, bs.unitRings[u].window(head: head, tapStep: unit.dilation), rows: rows, unit.conv1,
                              act: CodecActOutput(buffer: bs.branch, ld: c, channels: c, activation: unit.act2))
                }
                list.append { k.product($0, CodecWindow(bs.branch, ld: c), rows: rows, unit.conv2, main: stream, accumulate: true, act: next) }
            }
            stages["decoder\(i + 1)"] = StageView(buffer: bs.stream, offset: 0, rows: rows, width: c)
        }
        let samples = f * CodecShape.samplesPerFrame
        let finalHead = n0 * CodecShape.samplesPerFrame
        stages["decoder5"] = StageView(buffer: finalRing.buffer, offset: (finalHead % finalRing.rows) * finalRing.width * 4, rows: samples, width: finalRing.width)
        let pcmOffset = (steps % pcmSlots) * chunkFrames * CodecShape.samplesPerFrame * 4
        let pcm = self.pcm
        list.append {
            k.product($0, finalRing.window(head: finalHead), rows: samples, w.finalConv, act: CodecActOutput(buffer: pcm, offset: pcmOffset, ld: 1, channels: 1, activation: .clamp))
        }
        frames += f
        steps += 1
        lastStages = stages
        return (list, pcmOffset, samples)
    }

    /// Decodes a whole stream from the start, `chunkFrames` frames per command buffer.
    func decode(codes: [[Int]]) throws -> [Float] {
        let books = CodecShape.codebooks
        guard codes.allSatisfy({ $0.count == books && $0.allSatisfy { (0..<CodecShape.codebookSize).contains($0) } }) else {
            throw BreezeTTSError.badInput("codec frames are \(books) codes in 0..<\(CodecShape.codebookSize)")
        }
        guard codes.count <= maxFrames else { throw BreezeTTSError.badInput("\(codes.count) frames exceed the decoder's \(maxFrames)") }
        if batchCodes == nil { batchCodes = try ctx.makeBuffer(bytes: chunkFrames * books * 4, label: "codec.batchCodes") }
        if batchKernels == nil { batchKernels = try Kernels(ctx: ctx) }
        let buffer = batchCodes!, engine = batchKernels!
        reset()
        var out: [Float] = []
        out.reserveCapacity(codes.count * CodecShape.samplesPerFrame)
        var start = 0
        while start < codes.count {
            let f = min(chunkFrames, codes.count - start)
            let p = buffer.contents().bindMemory(to: UInt32.self, capacity: chunkFrames * books)
            for i in 0..<f {
                for q in 0..<books { p[i * books + q] = UInt32(codes[start + i][q]) }
            }
            let step = append(codes: buffer, codesOffset: 0, count: f)
            guard let cb = ctx.queue.makeCommandBuffer(), let ce = cb.makeComputeCommandEncoder() else { throw BreezeTTSError.gpuFailure("no command buffer") }
            let e = Encoder(k: engine, ce: ce)
            for d in step.dispatches { d(e) }
            ce.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let error = cb.error { throw BreezeTTSError.gpuFailure("\(error)") }
            out += pcm.floats(count: step.samples, offset: step.pcmOffset)
            start += f
        }
        return out
    }
}
