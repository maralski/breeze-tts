// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// The T5Gemma2 text encoder as Breeze configures it: bidirectional, sandwich norms with Gemma's
/// (1 + w) RMSNorm, q and k normed before RoPE, four query heads on one KV head.
struct TextEncoderConfig {
    let hidden = 1152
    let heads = 4
    let kvHeads = 1
    let headDim = 256
    let intermediate = 6912
    let layers = 26
    let eps: Float = 1e-6
    /// sqrt(1152) rounded through bf16, as the reference builds the buffer.
    let embedScale: Float = 34
    /// query_pre_attn_scalar 256.
    let attentionScale: Float = 1.0 / 16
    /// Query i of a sliding layer sees keys i - 255 ... i + 256.
    let window = (before: 255, after: 256)
    let fullAttentionLayers: Set<Int> = [5, 11, 17, 23]
    let slidingRope = RopeFrequencies.plain(theta: 1e4, dim: 256)
    let fullRope = RopeFrequencies.linear(theta: 1e6, dim: 256, factor: 8)
    /// `text_encoder_proj` output width, the backbone's.
    let projected = BreezeShape.embedWidth
    let eoiTokenID: Int32 = 256000

    var qWidth: Int { heads * headDim }
    var qkvWidth: Int { (heads + 2 * kvHeads) * headDim }

    static let t5gemma2 = TextEncoderConfig()
}

struct TextEncoderLayerWeights {
    let preAttentionNorm: GPUTensor
    let qkv: Linear
    /// q_norm then k_norm.
    let qkNorm: GPUTensor
    let o: Linear
    let postAttentionNorm: GPUTensor
    let preFeedForwardNorm: GPUTensor
    let gate: Linear
    let up: Linear
    let down: Linear
    let postFeedForwardNorm: GPUTensor
}

struct TextEncoderWeights {
    let layers: [TextEncoderLayerWeights]
    let finalNorm: GPUTensor
    let projection: Linear

    init(_ a: WeightArena, config c: TextEncoderConfig) throws {
        layers = try (0..<c.layers).map { i in
            let p = "text_encoder.layers.\(i)", at = "\(p).self_attn"
            return TextEncoderLayerWeights(
                preAttentionNorm: try a.vector("\(p).pre_self_attn_layernorm.weight", count: c.hidden),
                qkv: try a.stackedLinear(["\(at).q_proj", "\(at).k_proj", "\(at).v_proj"], n: [c.qWidth, c.headDim, c.headDim], k: c.hidden),
                qkNorm: try a.tensor("\(at).q_norm.weight", "\(at).k_norm.weight"),
                o: try a.linear("\(at).o_proj", n: c.hidden, k: c.qWidth),
                postAttentionNorm: try a.vector("\(p).post_self_attn_layernorm.weight", count: c.hidden),
                preFeedForwardNorm: try a.vector("\(p).pre_feedforward_layernorm.weight", count: c.hidden),
                gate: try a.linear("\(p).mlp.gate_proj", n: c.intermediate, k: c.hidden),
                up: try a.linear("\(p).mlp.up_proj", n: c.intermediate, k: c.hidden),
                down: try a.linear("\(p).mlp.down_proj", n: c.hidden, k: c.intermediate),
                postFeedForwardNorm: try a.vector("\(p).post_feedforward_layernorm.weight", count: c.hidden))
        }
        finalNorm = try a.vector("text_encoder.norm.weight", count: c.hidden)
        projection = try a.linear("text_encoder_proj", n: c.projected, k: c.hidden)
    }
}

/// The 262158 x 1152 embedding table stays in the mapped checkpoint: a request reads a few
/// hundred rows of it, gathered on the CPU, and the table would cost 600 MB of GPU memory.
struct TextEmbedding {
    static let table = "text_encoder.embed_tokens.weight"
    static let eoi = "text_encoder.embed_tokens.eoi_embedding"
    let store: any TensorStore
    let config: TextEncoderConfig

    init(_ store: any TensorStore, config: TextEncoderConfig) throws {
        let t = try store.entry(Self.table)
        guard t.dtype == "BF16", t.shape.count == 2, t.shape[1] == config.hidden else {
            throw BreezeTTSError.badCheckpoint("\(Self.table) is \(t.dtype) \(t.shape)")
        }
        self.store = store
        self.config = config
    }

    /// Writes `W[id] * 34` per id as fp32 rows of `hidden` floats at `destination`; the
    /// end-of-image id takes `eoi_embedding` instead, unscaled.
    func gather(_ ids: [Int32], into destination: UnsafeMutablePointer<Float>) throws {
        let width = config.hidden
        let eoi = ids.contains(config.eoiTokenID) ? try store.floats(Self.eoi) : []
        try store.withBytes(Self.table) { raw, entry in
            let vocab = entry.shape[0]
            for (r, id) in ids.enumerated() {
                let dst = destination + r * width
                if id == config.eoiTokenID {
                    for d in 0..<width { dst[d] = eoi[d] }
                    continue
                }
                guard id >= 0, Int(id) < vocab else { throw BreezeTTSError.badInput("token id \(id) is outside the text vocabulary") }
                let row = Int(id) * width * 2
                for d in 0..<width {
                    let bits = raw.loadUnaligned(fromByteOffset: row + 2 * d, as: UInt16.self)
                    dst[d] = Float(bitPattern: UInt32(bits) << 16) * config.embedScale
                }
            }
        }
    }
}

/// Scratch for one segment of up to `maxRows` rows; the residual stream lives in the caller's
/// buffer. Every buffer has 8 spare rows for the projection kernels' row chunks.
final class TextEncoderWorkspace {
    let maxRows: Int
    let normed, qkv, attn, mlp, delta: MTLBuffer
    let keys: KVCacheLayer

    init(ctx: MetalContext, config c: TextEncoderConfig, maxRows: Int) throws {
        self.maxRows = maxRows
        let rows = maxRows + 8
        normed = try ctx.makeBuffer(bytes: rows * c.hidden * 4, label: "text.normed")
        qkv = try ctx.makeBuffer(bytes: rows * c.qkvWidth * 4, label: "text.qkv")
        attn = try ctx.makeBuffer(bytes: rows * c.qWidth * 4, label: "text.attn")
        mlp = try ctx.makeBuffer(bytes: rows * c.intermediate * 4, label: "text.mlp")
        delta = try ctx.makeBuffer(bytes: rows * c.hidden * 4, label: "text.delta")
        let k = try ctx.makeBuffer(bytes: maxRows * c.headDim * 4, label: "text.k")
        let v = try ctx.makeBuffer(bytes: maxRows * c.headDim * 4, label: "text.v")
        keys = KVCacheLayer(k: GPUTensor(buffer: k, offset: 0, count: maxRows * c.headDim), v: GPUTensor(buffer: v, offset: 0, count: maxRows * c.headDim),
                            capacity: maxRows, seqStride: maxRows * c.headDim)
    }
}

/// Encodes one text segment: its rows at positions 0 ..< rows, attending to each other only.
struct TextEncoder {
    let config: TextEncoderConfig
    let weights: TextEncoderWeights
    let work: TextEncoderWorkspace
    let slidingFrequencies, fullFrequencies: MTLBuffer

    init(ctx: MetalContext, weights: TextEncoderWeights, config: TextEncoderConfig = .t5gemma2, maxRows: Int) throws {
        self.config = config
        self.weights = weights
        work = try TextEncoderWorkspace(ctx: ctx, config: config, maxRows: maxRows)
        slidingFrequencies = try ctx.makeBuffer(floats: config.slidingRope.values, label: "text.invFreq.sliding")
        fullFrequencies = try ctx.makeBuffer(floats: config.fullRope.values, label: "text.invFreq.full")
    }

    /// `x` holds the scaled embeddings on entry and the residual stream after the last layer on
    /// exit; `out` receives the projected rows. `afterLayer` is called once per layer, after its
    /// dispatches, and `afterNorm` after the final norm (into the workspace's `normed`), so a
    /// caller can encode copies of intermediate states.
    func encode(_ e: Encoder, x: ActView, rows: Int, out: ActView, afterLayer: ((Int) -> Void)? = nil, afterNorm: ((ActView) -> Void)? = nil) {
        let c = config
        precondition(rows > 0 && rows <= work.maxRows)
        let normed = ActView(work.normed, ld: c.hidden), qkv = ActView(work.qkv, ld: c.qkvWidth)
        let attn = ActView(work.attn, ld: c.qWidth), mlp = ActView(work.mlp, ld: c.intermediate)
        let delta = ActView(work.delta, ld: c.hidden)
        let block = RowBlock(rows: rows, positions: [0])
        for (i, layer) in weights.layers.enumerated() {
            let full = c.fullAttentionLayers.contains(i)
            e.rmsnorm(x, rows: rows, width: c.hidden, weight: layer.preAttentionNorm, normed, eps: c.eps, onePlusWeight: true)
            e.linear(normed, rows: rows, layer.qkv, qkv)
            e.normRopeKV(qkv, block: block, heads: c.heads, kvHeads: c.kvHeads, headDim: c.headDim, cache: work.keys,
                         invFreq: full ? fullFrequencies : slidingFrequencies, norm: .onePlusWeight(layer.qkNorm, eps: c.eps))
            e.attentionWindow(qkv, rows: rows, total: rows, window: full ? nil : c.window, keys: work.keys.k, values: work.keys.v, scale: c.attentionScale, attn)
            e.linear(attn, rows: rows, layer.o, delta)
            e.rmsnorm(delta, rows: rows, width: c.hidden, weight: layer.postAttentionNorm, x, eps: c.eps, onePlusWeight: true, accumulate: true)
            e.rmsnorm(x, rows: rows, width: c.hidden, weight: layer.preFeedForwardNorm, normed, eps: c.eps, onePlusWeight: true)
            e.linear(normed, rows: rows, layer.gate, mlp, flags: .geluGate, gate: layer.up)
            e.linear(mlp, rows: rows, layer.down, delta)
            e.rmsnorm(delta, rows: rows, width: c.hidden, weight: layer.postFeedForwardNorm, x, eps: c.eps, onePlusWeight: true, accumulate: true)
            afterLayer?(i)
        }
        e.rmsnorm(x, rows: rows, width: c.hidden, weight: weights.finalNorm, normed, eps: c.eps, onePlusWeight: true)
        afterNorm?(normed)
        e.linear(normed, rows: rows, weights.projection, out)
    }
}
