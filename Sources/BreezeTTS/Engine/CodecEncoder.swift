// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

struct CodecEncoderWeights {
    struct Stage {
        let channels: Int
        let ratio: Int
        /// Residual block: kernel 3 to half the channels, then kernel 1 back.
        let narrow: CodecConv
        let widen: CodecConv
        /// Kernel 2 * ratio, stride ratio, to twice the channels.
        let down: CodecConv
    }

    struct Layer {
        let norm1Weight, norm1Bias: GPUTensor
        /// q, k and v stacked: [1536][512].
        let qkv: CodecConv
        /// Scaled by the attention layer scale.
        let o: CodecConv
        let norm2Weight, norm2Bias: GPUTensor
        let fc1: CodecConv
        /// Scaled by the MLP layer scale.
        let fc2: CodecConv
    }

    static let hidden = 512
    static let heads = 8
    static let ratios = [4, 5, 6, 8]
    static let layerNormEps: Float = 1e-5

    let first: CodecConv
    let stages: [Stage]
    let last: CodecConv
    let layers: [Layer]
    let downsample: CodecConv
    /// The semantic and acoustic input projections stacked: [256 + 256][512].
    let quantizerProj: CodecConv
    /// [16][2048][256] and the squared norms [16][2048]: the semantic codebook, then the first
    /// 15 acoustic ones. The checkpoint's other 16 acoustic codebooks never affect these codes.
    let codebooks: GPUTensor
    let codebookNorms: GPUTensor

    init(_ l: CodecLoader) throws {
        let h = Self.hidden, e = "encoder.encoder.layers"
        first = try l.conv("\(e).0.conv", out: 64, in: 1, kernel: 7)
        var channels = 64
        var stages: [Stage] = []
        for (i, ratio) in Self.ratios.enumerated() {
            let block = "\(e).\(1 + 3 * i).block"
            stages.append(Stage(channels: channels, ratio: ratio,
                                narrow: try l.conv("\(block).1.conv", out: channels / 2, in: channels, kernel: 3),
                                widen: try l.conv("\(block).3.conv", out: channels, in: channels / 2, kernel: 1),
                                down: try l.conv("\(e).\(3 + 3 * i).conv", out: 2 * channels, in: channels, kernel: 2 * ratio)))
            channels *= 2
        }
        self.stages = stages
        last = try l.conv("\(e).14.conv", out: h, in: channels, kernel: 3)
        let t = "encoder.encoder_transformer.layers"
        layers = try (0..<8).map { i in
            let p = "\(t).\(i)", a = "\(p).self_attn"
            return Layer(norm1Weight: try l.vector("\(p).input_layernorm.weight", count: h), norm1Bias: try l.vector("\(p).input_layernorm.bias", count: h),
                         qkv: try l.linear(["\(a).q_proj.weight", "\(a).k_proj.weight", "\(a).v_proj.weight"], n: [h, h, h], k: h),
                         o: try l.linear(["\(a).o_proj.weight"], n: [h], k: h, rowScale: try l.floats("\(p).self_attn_layer_scale.scale", shape: [h])),
                         norm2Weight: try l.vector("\(p).post_attention_layernorm.weight", count: h), norm2Bias: try l.vector("\(p).post_attention_layernorm.bias", count: h),
                         fc1: try l.linear(["\(p).mlp.fc1.weight"], n: [4 * h], k: h),
                         fc2: try l.linear(["\(p).mlp.fc2.weight"], n: [h], k: 4 * h, rowScale: try l.floats("\(p).mlp_layer_scale.scale", shape: [h])))
        }
        downsample = try l.conv("encoder.downsample.conv", out: h, in: h, kernel: 4)
        let q = "encoder.quantizer", dim = CodecShape.codebookDim
        quantizerProj = try l.linear(["\(q).semantic_residual_vector_quantizer.input_proj.weight", "\(q).acoustic_residual_vector_quantizer.input_proj.weight"],
                                     n: [dim, dim], k: h, squeeze: true)
        var books = try l.codebook(sum: "\(q).semantic_residual_vector_quantizer.layers.0.codebook.embed_sum", usage: "\(q).semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage")
        for i in 0..<(CodecShape.codebooks - 1) {
            let c = "\(q).acoustic_residual_vector_quantizer.layers.\(i).codebook"
            books += try l.codebook(sum: "\(c).embed_sum", usage: "\(c).cluster_usage")
        }
        codebooks = try l.tensor(books, label: "encoder.codebooks")
        let norms = (0..<(books.count / dim)).map { c in
            var s: Float = 0
            for d in 0..<dim { s += books[c * dim + d] * books[c * dim + d] }
            return s
        }
        codebookNorms = try l.tensor(norms, label: "encoder.codebookNorms")
    }
}

/// The Qwen3-TTS 12 Hz tokenizer's encoder: 24 kHz mono samples to 16 codes per 1920 samples.
/// A causal conv stack to 25 Hz, an 8-layer causal transformer, a stride-2 conv to 12.5 Hz, and
/// residual vector quantization. Runs once per reference clip, the whole clip at a time.
final class CodecEncoder {
    let ctx: MetalContext
    let kernels: CodecKernels
    let weights: CodecEncoderWeights
    private let engine: Kernels

    /// The longest clip: the transformer attends over at most `CodecKernels.longWindow` 25 Hz frames.
    static let maxSamples = CodecKernels.longWindow * 960

    struct Stages {
        /// Conv stack output [frames25][512], transformer output [frames25][512], downsampled [frames][512].
        var seanet: [Float] = [], transformer: [Float] = [], downsample: [Float] = []
        var codes: [[Int16]] = []
    }

    convenience init(directory: URL, ctx: MetalContext) throws {
        try self.init(ctx: ctx, kernels: try CodecKernels(ctx: ctx), weights: try CodecEncoderWeights(try CodecLoader(directory: directory, ctx: ctx)))
    }

    init(ctx: MetalContext, kernels: CodecKernels, weights: CodecEncoderWeights, engine: Kernels? = nil) throws {
        self.ctx = ctx
        self.kernels = kernels
        self.weights = weights
        self.engine = try engine ?? Kernels(ctx: ctx)
    }

    /// Codes [frames][16] for mono samples at 24 kHz, frames = ceil(count / 1920).
    func encode(samples: [Float]) throws -> [[Int16]] { try encodeStages(samples: samples).codes }

    /// Codes for mono samples at any rate, resampled to 24 kHz first (see `codecResample`).
    func encode(samples: [Float], sampleRate: Int) throws -> [[Int16]] {
        guard sampleRate > 0 else { throw BreezeTTSError.badInput("sample rate must be positive") }
        return try encode(samples: codecResample(samples, from: sampleRate, to: CodecShape.sampleRate))
    }

    private func run(_ body: (Encoder) -> Void) throws {
        guard let cb = ctx.queue.makeCommandBuffer(), let ce = cb.makeComputeCommandEncoder() else { throw BreezeTTSError.gpuFailure("no command buffer") }
        body(Encoder(k: engine, ce: ce))
        ce.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { throw BreezeTTSError.gpuFailure("\(error)") }
    }

    private func plain(_ rows: Int, _ width: Int, _ label: String) throws -> MTLBuffer {
        try ctx.makeBuffer(bytes: (rows + CodecRing.slackRows) * width * 4, label: "encoder." + label)
    }

    /// The zero-padded input of a causal conv over `rows` rows: `kernel - stride` rows of left
    /// context, the rows, and zeros up to a whole number of strides.
    private struct Padded {
        let buffer: MTLBuffer
        let width: Int
        let pad: Int
        let stride: Int
        let outRows: Int

        init(ctx: MetalContext, rows: Int, width: Int, kernel: Int, stride: Int, label: String) throws {
            self.width = width
            self.stride = stride
            pad = kernel - stride
            outRows = (rows + stride - 1) / stride
            buffer = try ctx.makeBuffer(bytes: (pad + outRows * stride + CodecRing.slackRows * stride + kernel) * width * 4, label: "encoder." + label)
        }

        var window: CodecWindow { CodecWindow(buffer, ld: width, rowStep: stride) }
        func target(_ activation: CodecActivation) -> CodecActOutput {
            CodecActOutput(buffer: buffer, ld: width, channels: width, row0: pad, activation: activation)
        }
    }

    func encodeStages(samples: [Float]) throws -> Stages {
        guard !samples.isEmpty, samples.count <= Self.maxSamples else {
            throw BreezeTTSError.badInput("a reference clip is 1...\(Self.maxSamples) samples at 24 kHz, got \(samples.count)")
        }
        guard samples.allSatisfy(\.isFinite) else { throw BreezeTTSError.badInput("reference samples must be finite") }
        let k = kernels, w = weights, h = CodecEncoderWeights.hidden
        var stages = Stages()

        var rows = samples.count
        let input = try Padded(ctx: ctx, rows: rows, width: 1, kernel: 7, stride: 1, label: "input")
        input.buffer.write(samples, offset: input.pad * 4)
        var list: [(Encoder) -> Void] = []
        var stream = try plain(rows, 64, "stream0")
        var blockInput = try Padded(ctx: ctx, rows: rows, width: 64, kernel: 3, stride: 1, label: "block0")
        do {
            let s = stream, b = blockInput, n = rows
            list.append { k.product($0, input.window, rows: n, w.first, main: ActView(s, ld: 64), act: b.target(.elu)) }
        }
        var lastInput: Padded?
        for (i, stage) in w.stages.enumerated() {
            let c = stage.channels, n = rows, s = stream, b = blockInput
            let mid = try plain(n, c / 2, "mid\(i)")
            let down = try Padded(ctx: ctx, rows: n, width: c, kernel: 2 * stage.ratio, stride: stage.ratio, label: "down\(i)")
            list.append { k.product($0, b.window, rows: n, stage.narrow, act: CodecActOutput(buffer: mid, ld: c / 2, channels: c / 2, activation: .elu)) }
            list.append { k.product($0, CodecWindow(mid, ld: c / 2), rows: n, stage.widen, main: ActView(s, ld: c), accumulate: true, act: down.target(.elu)) }
            rows = down.outRows
            let m = rows
            if i + 1 < w.stages.count {
                let nextStream = try plain(m, 2 * c, "stream\(i + 1)")
                let nextBlock = try Padded(ctx: ctx, rows: m, width: 2 * c, kernel: 3, stride: 1, label: "block\(i + 1)")
                list.append { k.product($0, down.window, rows: m, stage.down, main: ActView(nextStream, ld: 2 * c), act: nextBlock.target(.elu)) }
                stream = nextStream
                blockInput = nextBlock
            } else {
                let tail = try Padded(ctx: ctx, rows: m, width: 2 * c, kernel: 3, stride: 1, label: "tail")
                list.append { k.product($0, down.window, rows: m, stage.down, act: tail.target(.elu)) }
                lastInput = tail
            }
        }
        let frames25 = rows
        let x = try plain(frames25, h, "x"), xView = ActView(x, ld: h)
        list.append { k.product($0, lastInput!.window, rows: frames25, w.last, main: xView) }
        try run { e in list.forEach { $0(e) } }
        stages.seanet = x.floats(count: frames25 * h)

        let xn = try plain(frames25, h, "xn"), qkv = try plain(frames25, 3 * h, "qkv"), attn = try plain(frames25, h, "attn"), wide = try plain(frames25, 4 * h, "wide")
        let heads = CodecEncoderWeights.heads, eps = CodecEncoderWeights.layerNormEps
        let keys = try ctx.makeBuffer(bytes: heads * frames25 * 64 * 4, label: "encoder.k"), values = try ctx.makeBuffer(bytes: heads * frames25 * 64 * 4, label: "encoder.v")
        let rope = try RopeTable(ctx: ctx, positions: frames25)
        let xnView = ActView(xn, ld: h), qkvView = ActView(qkv, ld: 3 * h), attnView = ActView(attn, ld: h)
        try run { e in
            for layer in w.layers {
                k.norm(e, xView, rows: frames25, width: h, weight: layer.norm1Weight, bias: layer.norm1Bias, xnView, eps: eps)
                k.product(e, CodecWindow(xnView), rows: frames25, layer.qkv, main: qkvView)
                k.ropeKV(e, qkvView, rows: frames25, heads: heads, pos0: 0, ring: frames25, k: keys, v: values, table: rope)
                k.attention(e, qkvView, rows: frames25, heads: heads, pos0: 0, window: frames25, ring: frames25, k: keys, v: values, attnView)
                k.product(e, CodecWindow(attnView), rows: frames25, layer.o, main: xView, accumulate: true)
                k.norm(e, xView, rows: frames25, width: h, weight: layer.norm2Weight, bias: layer.norm2Bias, xnView, eps: eps)
                k.product(e, CodecWindow(xnView), rows: frames25, layer.fc1, act: CodecActOutput(buffer: wide, ld: 4 * h, channels: 4 * h, activation: .gelu))
                k.product(e, CodecWindow(wide, ld: 4 * h), rows: frames25, layer.fc2, main: xView, accumulate: true)
            }
        }
        stages.transformer = x.floats(count: frames25 * h)

        // The downsampling conv pads by repeating the edge rows, unlike every other conv here.
        let frames = (frames25 + 1) / 2
        let rowAt = { (r: Int) in Array(stages.transformer[(r * h)..<((r + 1) * h)]) }
        var padded = rowAt(0) + rowAt(0) + stages.transformer
        for _ in 0..<(2 * frames - frames25) { padded += rowAt(frames25 - 1) }
        let downInput = try plain(padded.count / h + CodecRing.slackRows * 2, h, "downInput")
        downInput.write(padded)
        let down = try plain(frames, h, "down"), projected = try plain(frames, h, "projected")
        let codes = try ctx.makeBuffer(bytes: frames * CodecShape.codebooks * 4, label: "encoder.codes")
        try run { e in
            k.product(e, CodecWindow(downInput, ld: h, rowStep: 2), rows: frames, w.downsample, main: ActView(down, ld: h))
            self.quantize(e, ActView(down, ld: h), rows: frames, scratch: projected, codes: codes)
        }
        stages.downsample = down.floats(count: frames * h)
        stages.codes = Self.read(codes, rows: frames)
        return stages
    }

    /// Residual vector quantization of rows of the 12.5 Hz embedding: project into the semantic
    /// and the acoustic code space, pick the semantic code, then 15 acoustic codes greedily,
    /// each from the residual the previous ones left.
    func quantize(_ e: Encoder, _ embedding: ActView, rows: Int, scratch: MTLBuffer, codes: MTLBuffer) {
        let dim = CodecShape.codebookDim, size = CodecShape.codebookSize, w = weights
        let projected = ActView(scratch, ld: 2 * dim)
        kernels.product(e, CodecWindow(embedding), rows: rows, w.quantizerProj, main: projected)
        for q in 0..<CodecShape.codebooks {
            kernels.vqNearest(e, q == 0 ? projected : projected.column(dim), rows: rows,
                              codebook: GPUTensor(buffer: w.codebooks.buffer, offset: q * size * dim * 4, count: size * dim),
                              norms: GPUTensor(buffer: w.codebookNorms.buffer, offset: q * size * 4, count: size), codes: codes, slot: q)
        }
    }

    /// The codes for rows of a given 12.5 Hz embedding [rows][512]: the quantizer alone.
    func quantize(embedding: [Float]) throws -> [[Int16]] {
        let h = CodecEncoderWeights.hidden, rows = embedding.count / h
        let input = try plain(rows, h, "embedding"), scratch = try plain(rows, h, "projected")
        input.write(embedding)
        let codes = try ctx.makeBuffer(bytes: rows * CodecShape.codebooks * 4, label: "encoder.codes")
        try run { self.quantize($0, ActView(input, ld: h), rows: rows, scratch: scratch, codes: codes) }
        return Self.read(codes, rows: rows)
    }

    private static func read(_ codes: MTLBuffer, rows: Int) -> [[Int16]] {
        let p = codes.contents().bindMemory(to: UInt32.self, capacity: rows * CodecShape.codebooks)
        return (0..<rows).map { r in (0..<CodecShape.codebooks).map { Int16(p[r * CodecShape.codebooks + $0]) } }
    }
}
