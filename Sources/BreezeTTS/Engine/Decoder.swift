// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// The K and V caches of one decoder stack for `sequences` independent sequences, fp32, one
/// buffer for all layers. Each sequence keeps its own length, so the two sequences of a guided
/// generation can sit at different positions.
final class KVCache {
    let layers: [KVCacheLayer]
    let capacity: Int
    let sequences: Int
    /// Positions filled per sequence.
    private(set) var lengths: [Int]

    init(ctx: MetalContext, config: DecoderConfig, sequences: Int, label: String) throws {
        precondition(sequences >= 1 && sequences <= RowBlock.maxSequences)
        let capacity = config.capacity
        self.capacity = capacity
        self.sequences = sequences
        lengths = [Int](repeating: 0, count: sequences)
        let seqStride = config.kvHeads * capacity * config.headDim
        let perTensor = sequences * seqStride * 4
        let aligned = (perTensor + 255) & ~255
        let buffer = try ctx.makeBuffer(bytes: aligned * 2 * config.layers, label: label)
        layers = (0..<config.layers).map { i in
            KVCacheLayer(k: GPUTensor(buffer: buffer, offset: (2 * i) * aligned, count: perTensor / 4),
                         v: GPUTensor(buffer: buffer, offset: (2 * i + 1) * aligned, count: perTensor / 4),
                         capacity: capacity, seqStride: seqStride)
        }
    }

    func reset() { lengths = [Int](repeating: 0, count: sequences) }
    func setLengths(_ l: [Int]) {
        precondition(l.count == sequences && l.allSatisfy { $0 >= 0 && $0 <= capacity })
        lengths = l
    }
    func advance(_ rows: Int, sequences range: Range<Int>) {
        for s in range { lengths[s] += rows }
    }
}

/// Scratch activations for a decoder forward over at most `maxRows` rows in all. Every buffer
/// has 8 spare rows because the projection kernels read whole row chunks.
final class DecoderWorkspace {
    let maxRows: Int
    let normed, qkv, attn, mlp: MTLBuffer

    init(ctx: MetalContext, config: DecoderConfig, maxRows: Int, label: String) throws {
        self.maxRows = maxRows
        let rows = maxRows + 8
        normed = try ctx.makeBuffer(bytes: rows * config.hidden * 4, label: "\(label).normed")
        qkv = try ctx.makeBuffer(bytes: rows * config.qkvWidth * 4, label: "\(label).qkv")
        attn = try ctx.makeBuffer(bytes: rows * config.qWidth * 4, label: "\(label).attn")
        mlp = try ctx.makeBuffer(bytes: rows * config.intermediate * 4, label: "\(label).mlp")
    }
}

/// Encodes a pre-norm causal decoder stack (the backbone, or the depth decoder, by its config):
/// per layer `h += o(attn(norm1(h)))`, `h += down(silu(gate(norm2(h))) * up(norm2(h)))`.
struct DecoderForward {
    let weights: DecoderWeights
    let cache: KVCache
    let work: DecoderWorkspace
    let invFreq: MTLBuffer

    init(ctx: MetalContext, weights: DecoderWeights, cache: KVCache, work: DecoderWorkspace) throws {
        precondition(weights.config.rope.headDim == weights.config.headDim)
        self.weights = weights
        self.cache = cache
        self.work = work
        invFreq = try ctx.makeBuffer(floats: weights.config.rope.values, label: "invFreq")
    }

    /// Dispatches per layer.
    var layerDispatches: Int { weights.config.attention == .cached ? 8 : 7 }

    /// Runs the layers over `rows` new rows of each sequence in `sequences`, at the positions the
    /// cache holds for them. `x` holds the rows sequence-major on entry and the residual stream
    /// on exit; the final norm of each sequence's last row goes to consecutive rows of `out`.
    /// Advances the cache. `afterLayer` is called after each layer's dispatches, for a caller
    /// that encodes copies of the residual stream.
    func encode(_ e: Encoder, x: ActView, rows: Int, sequences: Range<Int>? = nil, out: ActView, afterLayer: ((Int) -> Void)? = nil) {
        let c = weights.config
        let range = sequences ?? 0..<cache.sequences
        let block = RowBlock(rows: rows, positions: range.map { cache.lengths[$0] })
        precondition(block.count <= work.maxRows && block.positions.max()! + rows <= cache.capacity)
        let normed = ActView(work.normed, ld: c.hidden)
        let qkv = ActView(work.qkv, ld: c.qkvWidth)
        let attn = ActView(work.attn, ld: c.qWidth)
        let mlp = ActView(work.mlp, ld: c.intermediate)
        for (i, layer) in weights.layers.enumerated() {
            let kv = cache.layers[i].sequences(from: range.lowerBound)
            e.rmsnorm(x, rows: block.count, width: c.hidden, weight: layer.norm1, normed, eps: c.eps)
            e.linear(normed, rows: block.count, layer.qkv, qkv)
            switch c.attention {
            case .cached:
                let norm: QKNorm = layer.qkNorm.map { .weight($0, eps: c.eps) } ?? .none
                e.normRopeKV(qkv, block: block, heads: c.heads, kvHeads: c.kvHeads, headDim: c.headDim, cache: kv, invFreq: invFreq, norm: norm)
                e.attentionCausal(qkv, block: block, heads: c.heads, kvHeads: c.kvHeads, cache: kv, attn)
            case .fusedShort:
                precondition(layer.qkNorm == nil && Set(block.positions).count == 1, "the short attention has no q/k norm and one position for all sequences")
                e.attentionShort(qkv, sequences: block.sequences, rows: rows, pos0: block.positions[0], heads: c.heads, kvHeads: c.kvHeads, cache: kv, invFreq: invFreq, attn)
            }
            e.linear(attn, rows: block.count, layer.o, x, flags: .accumulate)
            e.rmsnorm(x, rows: block.count, width: c.hidden, weight: layer.norm2, normed, eps: c.eps)
            e.linear(normed, rows: block.count, layer.gate, mlp, flags: .siluGate, gate: layer.up.w)
            e.linear(mlp, rows: block.count, layer.down, x, flags: .accumulate)
            afterLayer?(i)
        }
        e.rmsnorm(x.row(rows - 1).strided(rows * x.ld), rows: block.sequences, width: c.hidden, weight: weights.finalNorm, out, eps: c.eps)
        cache.advance(rows, sequences: range)
    }
}
