// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Copies the bf16 tensors one module needs into a single owned GPU buffer, each 256-byte aligned.
/// The builder runs twice: once to size the buffer, once to copy.
///
/// Owned copies, not the mapped shards in place: 423 tensors of shard 1 start at a file offset
/// that is 2 mod 16, so a kernel could only read them through 2-byte loads, and the q, k and v
/// projections could not be one weight.
final class WeightArena {
    private let store: any TensorStore
    private var planning = true
    private var cursor = 0
    private(set) var buffer: MTLBuffer?

    private init(store: any TensorStore) { self.store = store }

    static func load<T>(_ store: any TensorStore, ctx: MetalContext, label: String, _ build: (WeightArena) throws -> T) throws -> T {
        let arena = WeightArena(store: store)
        _ = try build(arena)
        arena.buffer = try ctx.makeBuffer(bytes: arena.cursor, label: label)
        arena.planning = false
        arena.cursor = 0
        return try build(arena)
    }

    private func place(bytes: Int) -> Int {
        let offset = cursor
        cursor += (bytes + 255) & ~255
        return offset
    }

    private func tensorAt(_ offset: Int, count: Int) -> GPUTensor {
        GPUTensor(buffer: buffer ?? placeholder, offset: offset, count: count)
    }

    /// A bf16 tensor, or several concatenated along their first axis (q, k, v rows for instance).
    func tensor(_ names: String...) throws -> GPUTensor {
        var total = 0
        for name in names {
            let e = try store.entry(name)
            guard e.dtype == "BF16" else { throw BreezeTTSError.badCheckpoint("\(name) is \(e.dtype), expected BF16") }
            total += e.byteRange.count
        }
        let offset = place(bytes: total)
        if let buffer, !planning {
            var dst = buffer.contents() + offset
            for name in names {
                try store.withBytes(name) { raw, _ in
                    memcpy(dst, raw.baseAddress!, raw.count)
                    dst += raw.count
                }
            }
        }
        return tensorAt(offset, count: total / 2)
    }

    private func expect(_ name: String, shape: [Int]) throws {
        let e = try store.entry(name)
        guard e.dtype == "BF16", e.shape == shape else {
            throw BreezeTTSError.badCheckpoint("\(name) is \(e.dtype) \(e.shape), expected BF16 \(shape)")
        }
    }

    func linear(_ prefix: String, n: Int, k: Int) throws -> Linear {
        try expect(prefix + ".weight", shape: [n, k])
        return Linear(w: try tensor(prefix + ".weight"), bias: nil, n: n, k: k)
    }

    /// The projection in `format`. The int8 forms are computed here, from the mapped bf16
    /// straight into the buffer, and exist nowhere else.
    func linear(_ prefix: String, n: Int, k: Int, format: DepthWeightFormat) throws -> Linear {
        let name = prefix + ".weight"
        switch format {
        case .bf16:
            return try linear(prefix, n: n, k: k)
        case .int8:
            try expect(name, shape: [n, k])
            let weights = place(bytes: n * k), scales = place(bytes: n * Int8Stripes.lanes * 2)
            if let buffer, !planning {
                try store.withBytes(name) { raw, _ in
                    Int8Stripes.quantize(raw.baseAddress!.assumingMemoryBound(to: UInt16.self), n: n, k: k,
                                         q: (buffer.contents() + weights).assumingMemoryBound(to: Int8.self),
                                         scales: (buffer.contents() + scales).assumingMemoryBound(to: Float16.self))
                }
            }
            return Linear(w: tensorAt(weights, count: n * k), bias: nil, n: n, k: k, scales: tensorAt(scales, count: n * Int8Stripes.lanes))
        case .int8Reference:
            try expect(name, shape: [n, k])
            let offset = place(bytes: n * k * 2)
            if let buffer, !planning {
                try store.withBytes(name) { raw, _ in
                    Int8Stripes.reference(raw.baseAddress!.assumingMemoryBound(to: UInt16.self), n: n, k: k,
                                          into: (buffer.contents() + offset).assumingMemoryBound(to: UInt16.self))
                }
            }
            return Linear(w: tensorAt(offset, count: n * k), bias: nil, n: n, k: k)
        }
    }

    /// Several [n_i, k] projections of one input as a single [sum n_i, k] weight.
    func stackedLinear(_ prefixes: [String], n: [Int], k: Int) throws -> Linear {
        var total = 0
        for (prefix, rows) in zip(prefixes, n) {
            try expect(prefix + ".weight", shape: [rows, k])
            total += rows * k * 2
        }
        let offset = place(bytes: total)
        if let buffer, !planning {
            var dst = buffer.contents() + offset
            for prefix in prefixes {
                try store.withBytes(prefix + ".weight") { raw, _ in
                    memcpy(dst, raw.baseAddress!, raw.count)
                    dst += raw.count
                }
            }
        }
        return Linear(w: tensorAt(offset, count: total / 2), bias: nil, n: n.reduce(0, +), k: k)
    }

    func vector(_ name: String, count: Int) throws -> GPUTensor {
        try expect(name, shape: [count])
        return try tensor(name)
    }

    /// A bf16 [count, inWidth, outWidth] stack of heads stored input-major, as `count` ordinary
    /// [outWidth, inWidth] weights. The checkpoint keeps the codebook heads transposed relative to
    /// every other projection; laid out row-major per output they stream like any other gemv.
    func transposedHeads(_ name: String, count: Int, inWidth: Int, outWidth: Int) throws -> [Linear] {
        try expect(name, shape: [count, inWidth, outWidth])
        let offsets = (0..<count).map { _ in place(bytes: outWidth * inWidth * 2) }
        if let buffer, !planning {
            try store.withBytes(name) { raw, _ in
                let src = raw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                for (i, offset) in offsets.enumerated() {
                    let dst = (buffer.contents() + offset).assumingMemoryBound(to: UInt16.self)
                    let head = src + i * inWidth * outWidth
                    for k in 0..<inWidth {
                        let row = head + k * outWidth
                        for n in 0..<outWidth { dst[n * inWidth + k] = row[n] }
                    }
                }
            }
        }
        return offsets.map { Linear(w: tensorAt($0, count: outWidth * inWidth), bias: nil, n: outWidth, k: inWidth) }
    }

    /// Stand-in during the sizing pass; never bound to an encoder.
    private var placeholder: MTLBuffer {
        struct Holder { nonisolated(unsafe) static let buffer = MTLCreateSystemDefaultDevice()!.makeBuffer(length: 16)! }
        return Holder.buffer
    }
}

/// The inverse frequencies of one RoPE table: `dim / 2` values, angle = position * value.
/// The model uses three recipes; kernels take whatever table the CPU supplies.
struct RopeFrequencies: Equatable {
    let values: [Float]
    var headDim: Int { values.count * 2 }

    /// inv[i] = theta^(-2i / dim): the backbone (theta 1e6, dim 128) and the encoder's sliding
    /// layers (theta 1e4, dim 256).
    static func plain(theta: Float, dim: Int) -> RopeFrequencies {
        RopeFrequencies(values: (0..<(dim / 2)).map { 1 / powf(theta, Float(2 * $0) / Float(dim)) })
    }

    /// The encoder's full-attention layers: the plain table divided by `factor`.
    static func linear(theta: Float, dim: Int, factor: Float) -> RopeFrequencies {
        RopeFrequencies(values: plain(theta: theta, dim: dim).values.map { $0 / factor })
    }

    /// transformers' llama3 scaling, as the depth decoder configures it (theta 5e5, dim 128,
    /// factor 32, low 1/512, high 1/128, original context 16).
    static func llama3(theta: Float, dim: Int, factor: Float, lowFreqFactor: Float, highFreqFactor: Float, originalContext: Float) -> RopeFrequencies {
        let lowWavelen = originalContext / lowFreqFactor, highWavelen = originalContext / highFreqFactor
        return RopeFrequencies(values: plain(theta: theta, dim: dim).values.map { inv in
            let wavelen = 2 * Float.pi / inv
            let scaled = wavelen > lowWavelen ? inv / factor : inv
            guard wavelen >= highWavelen, wavelen <= lowWavelen else { return scaled }
            let smooth = (originalContext / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
            return (1 - smooth) * scaled / factor + smooth * scaled
        })
    }
}

/// How a decoder layer attends.
enum DecoderAttention: Equatable {
    /// Keys go to the cache in one dispatch and a second attends over it: any cache length.
    case cached
    /// One dispatch rotates, writes the cache and attends, with every score in registers: the
    /// depth decoder's sequences of at most `AttentionLimits.shortKeys` positions.
    case fusedShort
}

/// One pre-norm causal decoder stack (RMSNorm, GQA attention, SwiGLU), the shape the backbone
/// and the depth decoder share.
struct DecoderConfig: Equatable {
    var hidden: Int
    var heads: Int
    var kvHeads: Int
    var headDim: Int
    var intermediate: Int
    var layers: Int
    var eps: Float
    /// Per-head RMSNorm of q and k before RoPE (Qwen3).
    var qkNorm: Bool
    var rope: RopeFrequencies
    var attention: DecoderAttention
    /// Positions one sequence's KV cache holds.
    var capacity: Int

    var qWidth: Int { heads * headDim }
    var kvWidth: Int { kvHeads * headDim }
    var qkvWidth: Int { qWidth + 2 * kvWidth }
    var attentionScale: Float { 1 / Float(headDim).squareRoot() }

    static let backbone = DecoderConfig(hidden: 2048, heads: 16, kvHeads: 8, headDim: 128, intermediate: 6144, layers: 28, eps: 1e-6,
                                        qkNorm: true, rope: .plain(theta: 1e6, dim: 128), attention: .cached, capacity: 2048)
    static let depth = DecoderConfig(hidden: 1024, heads: 8, kvHeads: 2, headDim: 128, intermediate: 8192, layers: 12, eps: 1e-5,
                                     qkNorm: false,
                                     rope: .llama3(theta: 5e5, dim: 128, factor: 32, lowFreqFactor: 0.001953125, highFreqFactor: 0.0078125, originalContext: 16),
                                     attention: .fusedShort, capacity: 17)
}

struct DecoderLayerWeights {
    let norm1: GPUTensor
    /// q, k and v as one [qWidth + 2 kvWidth, hidden] weight.
    let qkv: Linear
    /// q_norm then k_norm, `headDim` values each, when the stack has them.
    let qkNorm: GPUTensor?
    let o: Linear
    let norm2: GPUTensor
    let gate: Linear
    let up: Linear
    let down: Linear
}

struct DecoderWeights {
    let config: DecoderConfig
    let layers: [DecoderLayerWeights]
    let finalNorm: GPUTensor

    init(config: DecoderConfig, layers: [DecoderLayerWeights], finalNorm: GPUTensor) {
        self.config = config
        self.layers = layers
        self.finalNorm = finalNorm
    }

    /// `mlp` is the format of gate, up and down; everything else stays bf16.
    init(_ a: WeightArena, prefix: String, config c: DecoderConfig, mlp: DepthWeightFormat = .bf16) throws {
        config = c
        layers = try (0..<c.layers).map { i in
            let p = "\(prefix).layers.\(i)", at = "\(p).self_attn"
            return DecoderLayerWeights(
                norm1: try a.vector("\(p).input_layernorm.weight", count: c.hidden),
                qkv: try a.stackedLinear(["\(at).q_proj", "\(at).k_proj", "\(at).v_proj"], n: [c.qWidth, c.kvWidth, c.kvWidth], k: c.hidden),
                qkNorm: c.qkNorm ? try a.tensor("\(at).q_norm.weight", "\(at).k_norm.weight") : nil,
                o: try a.linear("\(at).o_proj", n: c.hidden, k: c.qWidth),
                norm2: try a.vector("\(p).post_attention_layernorm.weight", count: c.hidden),
                gate: try a.linear("\(p).mlp.gate_proj", n: c.intermediate, k: c.hidden, format: mlp),
                up: try a.linear("\(p).mlp.up_proj", n: c.intermediate, k: c.hidden, format: mlp),
                down: try a.linear("\(p).mlp.down_proj", n: c.hidden, k: c.intermediate, format: mlp))
        }
        finalNorm = try a.vector("\(prefix).norm.weight", count: c.hidden)
    }
}

/// Sizes the frame loop shares between the backbone, the depth decoder and the samplers.
enum BreezeShape {
    static let codebooks = 16
    /// Rows of one codebook in the shared embedding table: 2048 codes and three reserved ids.
    static let codebookVocab = 2051
    /// Backbone classes: the codebook vocabulary and the end-of-speech class 2051.
    static let backboneVocab = 2052
    static let endOfSpeech = 2051
    static let reservedIDs = 2048..<2051
    static let embedWidth = 2048
}

/// Everything the frame loop reads: the backbone, its head, the depth decoder with its input
/// projector and per-codebook heads, and the embedding table both stacks share.
/// `embed_text_tokens` and `codec_model.*` are in the checkpoint and never read by inference.
struct FrameWeights {
    let backbone: DecoderWeights
    /// [2052, 2048].
    let lmHead: Linear
    let depth: DecoderWeights
    /// [1024, 2048], applied to the backbone hidden state and to every code embedding.
    let depthProjector: Linear
    /// Head i scores codebook i + 1; each is [2051, 1024] after the load-time transpose.
    let codebookHeads: [Linear]
    /// bf16 [16 * 2051, 2048]: row `code + q * 2051` embeds code `code` of codebook `q`.
    let embedding: GPUTensor

    init(backbone: DecoderWeights, lmHead: Linear, depth: DecoderWeights, depthProjector: Linear, codebookHeads: [Linear], embedding: GPUTensor) {
        self.backbone = backbone
        self.lmHead = lmHead
        self.depth = depth
        self.depthProjector = depthProjector
        self.codebookHeads = codebookHeads
        self.embedding = embedding
    }

    /// `depth` is the format of the depth decoder's MLP weights.
    /// The same weights with the depth decoder's layers loaded again in `format`, for a bench
    /// that compares formats in one process.
    func withDepth(_ format: DepthWeightFormat, store: any TensorStore, ctx: MetalContext) throws -> FrameWeights {
        let stack = try WeightArena.load(store, ctx: ctx, label: "weights.depth.\(format)") {
            try DecoderWeights($0, prefix: "depth_decoder.model", config: .depth, mlp: format)
        }
        return FrameWeights(backbone: backbone, lmHead: lmHead, depth: stack, depthProjector: depthProjector, codebookHeads: codebookHeads, embedding: embedding)
    }

    init(_ store: any TensorStore, ctx: MetalContext, depth format: DepthWeightFormat = .bf16) throws {
        backbone = try WeightArena.load(store, ctx: ctx, label: "weights.backbone") {
            try DecoderWeights($0, prefix: "backbone_model", config: .backbone)
        }
        (lmHead, depth, depthProjector, codebookHeads, embedding) = try WeightArena.load(store, ctx: ctx, label: "weights.depth") { a in
            let d = DecoderConfig.depth
            let lm = try a.linear("lm_head", n: BreezeShape.backboneVocab, k: DecoderConfig.backbone.hidden)
            let stack = try DecoderWeights(a, prefix: "depth_decoder.model", config: d, mlp: format)
            let projector = try a.linear("depth_decoder.model.inputs_embeds_projector", n: d.hidden, k: BreezeShape.embedWidth)
            let heads = try a.transposedHeads("depth_decoder.codebooks_head.weight", count: BreezeShape.codebooks - 1, inWidth: d.hidden, outWidth: BreezeShape.codebookVocab)
            let table = try a.linear("depth_decoder.model.embed_tokens", n: BreezeShape.codebooks * BreezeShape.codebookVocab, k: BreezeShape.embedWidth)
            return (lm, stack, projector, heads, table.w)
        }
    }
}
