// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// fp32 weights of one convolution laid out as a matrix: [n rounded up to `rowQuantum`][taps * cin],
/// row-major, zero rows past n. Column `j * cin + c` multiplies channel c of tap j.
struct CodecConv {
    let w: GPUTensor
    let bias: GPUTensor?
    let n: Int
    let cin: Int
    let taps: Int
    var k: Int { cin * taps }

    /// Weight rows are padded to this, the widest tile block a simdgroup loads.
    static let rowQuantum = 64
}

/// Where a product reads its input: time rows `ld` floats apart starting `offset` bytes into
/// `buffer`. Output row m reads tap j from time row `m * rowStep + j * tapStep`.
struct CodecWindow {
    let buffer: MTLBuffer
    let offset: Int
    let ld: Int
    var rowStep = 1
    var tapStep = 1

    init(_ buffer: MTLBuffer, offset: Int = 0, ld: Int, rowStep: Int = 1, tapStep: Int = 1) {
        self.buffer = buffer
        self.offset = offset
        self.ld = ld
        self.rowStep = rowStep
        self.tapStep = tapStep
    }

    init(_ view: ActView) { self.init(view.buffer, offset: view.offset, ld: view.ld) }
}

enum CodecActivation {
    case identity
    /// x + invB * sin(a x)^2 per channel, with a = exp(alpha) and invB = 1 / (exp(beta) + 1e-9).
    case snake(a: GPUTensor, invB: GPUTensor)
    case elu
    /// The erf form.
    case gelu
    case clamp
}

/// Where a product stores the activated copy of its output, as time rows of `channels` floats.
/// `ringRows` 0 is a plain buffer; otherwise the rows wrap at `ringRows` and are stored twice.
struct CodecActOutput {
    let buffer: MTLBuffer
    var offset = 0
    let ld: Int
    let channels: Int
    var ringRows = 0
    var row0 = 0
    let activation: CodecActivation
}

/// The input history of one causal conv: the `pad` time rows of left context and the rows of
/// the step in progress. Row r of the stream lives at ring rows `r % rows` and `r % rows + rows`,
/// so the window a step reads is contiguous and nothing is ever shifted.
struct CodecRing {
    let buffer: MTLBuffer
    let width: Int
    let pad: Int
    /// Ring period: the context plus the most rows one step adds.
    let rows: Int

    /// Rows past the valid region a tile kernel may load (and ignore).
    static let slackRows = 32

    init(ctx: MetalContext, width: Int, pad: Int, maxNewRows: Int, label: String) throws {
        self.width = width
        self.pad = pad
        rows = pad + maxNewRows
        buffer = try ctx.makeBuffer(bytes: (2 * rows + Self.slackRows) * width * 4, label: label)
    }

    /// The window of a step that starts after `head` rows of the stream: context first.
    func window(head: Int, tapStep: Int = 1) -> CodecWindow {
        let base = ((head - pad) % rows + rows) % rows
        return CodecWindow(buffer, offset: base * width * 4, ld: width, tapStep: tapStep)
    }

    func output(head: Int, _ activation: CodecActivation) -> CodecActOutput {
        CodecActOutput(buffer: buffer, ld: width, channels: width, ringRows: rows, row0: head % rows, activation: activation)
    }

    /// Bytes of ring storage a reset must zero.
    var storedBytes: Int { 2 * rows * width * 4 }
}

/// The codec's pipelines and their dispatch wrappers. Every method appends one dispatch to `e`.
final class CodecKernels: @unchecked Sendable {
    struct Tile: Hashable { let tm, tn: Int }
    static let tiles = [Tile(tm: 1, tn: 2), Tile(tm: 1, tn: 4), Tile(tm: 1, tn: 8), Tile(tm: 2, tn: 2), Tile(tm: 2, tn: 4), Tile(tm: 4, tn: 2), Tile(tm: 4, tn: 4)]
    let gemm: [Tile: MTLComputePipelineState]
    let direct, norm, dwconvLN, swiglu, zero, rvqSum, vqNearest, ropeKV, attentionShort, attentionLong: MTLComputePipelineState

    /// Keys one query of `attentionShort` can see.
    static let shortWindow = 96
    /// Keys one query of the long attention can see: the encoder's limit on clip length.
    static let longWindow = 1024

    init(ctx: MetalContext) throws {
        var g: [Tile: MTLComputePipelineState] = [:]
        for t in Self.tiles { g[t] = try ctx.pipeline("cgemm_\(t.tm)_\(t.tn)") }
        gemm = g
        direct = try ctx.pipeline("cconv_direct")
        norm = try ctx.pipeline("codec_norm")
        dwconvLN = try ctx.pipeline("codec_dwconv_ln")
        swiglu = try ctx.pipeline("codec_swiglu")
        zero = try ctx.pipeline("codec_zero")
        rvqSum = try ctx.pipeline("codec_rvq_sum")
        vqNearest = try ctx.pipeline("codec_vq_nearest")
        ropeKV = try ctx.pipeline("codec_rope_kv")
        attentionShort = try ctx.pipeline("codec_attention_96")
        attentionLong = try ctx.pipeline("codec_attention_1024")
    }

    /// Forces one tile for every product: the bench's knob.
    nonisolated(unsafe) static var tileOverride: Tile?
    /// Simdgroups per threadgroup of the tile kernel. Overridable for the bench.
    nonisolated(unsafe) static var simdgroups = 4

    /// The tile for a product of `m` rows: tall blocks only when there are rows to fill them.
    static func tile(rows m: Int, n: Int) -> Tile {
        if let t = tileOverride { return t }
        if m <= 8 { return Tile(tm: 1, tn: 4) }
        if m <= 16 { return Tile(tm: 2, tn: 4) }
        return Tile(tm: 4, tn: 4)
    }

    private func setBytes<T>(_ e: Encoder, _ value: T, index: Int) {
        withUnsafeBytes(of: value) { e.ce.setBytes($0.baseAddress!, length: $0.count, index: index) }
    }

    /// One conv, transposed conv or linear layer over `m` output rows (see `cgemm_kernel`).
    /// `main` receives the raw result, added to what it holds with `accumulate`; `act` the
    /// activated copy.
    func product(_ e: Encoder, _ x: CodecWindow, rows m: Int, _ conv: CodecConv, main: ActView? = nil, accumulate: Bool = false, act: CodecActOutput? = nil) {
        struct P { var M, N, cin, taps, xld, rowStep, tapStep, yld, flags, sg, zRows, zRow0, zc, zld: UInt32 }
        precondition(main != nil || act != nil, "a product needs an output")
        precondition(!accumulate || main != nil, "accumulate adds to the main output")
        var flags: UInt32 = conv.bias == nil ? 0 : 1
        if accumulate { flags |= 2 }
        if main != nil { flags |= 4 }
        var actA: GPUTensor?, actB: GPUTensor?
        if let act {
            precondition(conv.n % act.channels == 0)
            flags |= 8
            switch act.activation {
            case .identity: break
            case .snake(let a, let invB): flags |= 16; (actA, actB) = (a, invB)
            case .elu: flags |= 32
            case .gelu: flags |= 64
            case .clamp: flags |= 128
            }
        }
        let tiled = conv.cin % 8 == 0 && conv.n >= 8
        let tile = Self.tile(rows: m, n: conv.n)
        let blocks = (conv.n + 8 * tile.tn - 1) / (8 * tile.tn)
        let sg = min(Self.simdgroups, blocks)
        e.start(tiled ? gemm[tile]! : direct, weightBytes: conv.n * conv.k * 4)
        let ce = e.ce
        ce.setBuffer(x.buffer, offset: x.offset, index: 0)
        ce.setBuffer(conv.w.buffer, offset: conv.w.offset, index: 1)
        let b = conv.bias ?? conv.w
        ce.setBuffer(b.buffer, offset: b.offset, index: 2)
        ce.setBuffer(main?.buffer ?? conv.w.buffer, offset: main?.offset ?? conv.w.offset, index: 3)
        ce.setBuffer(act?.buffer ?? conv.w.buffer, offset: act?.offset ?? conv.w.offset, index: 4)
        let a = actA ?? conv.w, ib = actB ?? conv.w
        ce.setBuffer(a.buffer, offset: a.offset, index: 5)
        ce.setBuffer(ib.buffer, offset: ib.offset, index: 6)
        setBytes(e, P(M: UInt32(m), N: UInt32(conv.n), cin: UInt32(conv.cin), taps: UInt32(conv.taps), xld: UInt32(x.ld), rowStep: UInt32(x.rowStep), tapStep: UInt32(x.tapStep),
                      yld: UInt32(main?.ld ?? 0), flags: flags, sg: UInt32(sg), zRows: UInt32(act?.ringRows ?? 0), zRow0: UInt32(act?.row0 ?? 0),
                      zc: UInt32(act?.channels ?? 1), zld: UInt32(act?.ld ?? 0)), index: 7)
        if tiled {
            ce.dispatchThreadgroups(MTLSize(width: (blocks + sg - 1) / sg, height: (m + 8 * tile.tm - 1) / (8 * tile.tm), depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 32 * sg, height: 1, depth: 1))
        } else {
            let across = min(conv.n, 32)
            ce.dispatchThreads(MTLSize(width: conv.n, height: m, depth: 1), threadsPerThreadgroup: MTLSize(width: across, height: max(1, 32 / across), depth: 1))
        }
    }

    /// RMSNorm (`bias` nil) or LayerNorm of `rows` rows of `width` floats.
    func norm(_ e: Encoder, _ x: ActView, rows: Int, width: Int, weight: GPUTensor, bias: GPUTensor? = nil, _ y: ActView, eps: Float) {
        struct P { var K, ldx, ldy: UInt32; var eps: Float; var mode: UInt32 }
        precondition(width % 4 == 0)
        e.start(norm, weightBytes: width * 4)
        let ce = e.ce
        ce.setBuffer(x.buffer, offset: x.offset, index: 0)
        ce.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        let b = bias ?? weight
        ce.setBuffer(b.buffer, offset: b.offset, index: 2)
        ce.setBuffer(y.buffer, offset: y.offset, index: 3)
        setBytes(e, P(K: UInt32(width), ldx: UInt32(x.ld), ldy: UInt32(y.ld), eps: eps, mode: bias == nil ? 0 : 1), index: 4)
        ce.dispatchThreads(MTLSize(width: rows * 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Causal depthwise conv then LayerNorm over channels (the ConvNeXt block's front).
    /// `w` is [channels][taps].
    func dwconvLN(_ e: Encoder, _ x: CodecWindow, rows: Int, channels: Int, taps: Int, w: GPUTensor, bias: GPUTensor, lnWeight: GPUTensor, lnBias: GPUTensor, _ y: ActView, eps: Float) {
        struct P { var C, taps, xld, ldy: UInt32; var eps: Float }
        precondition(x.rowStep == 1 && x.tapStep == 1)
        e.start(dwconvLN, weightBytes: channels * (taps + 3) * 4)
        let ce = e.ce
        ce.setBuffer(x.buffer, offset: x.offset, index: 0)
        for (i, t) in [w, bias, lnWeight, lnBias].enumerated() { ce.setBuffer(t.buffer, offset: t.offset, index: 1 + i) }
        ce.setBuffer(y.buffer, offset: y.offset, index: 5)
        setBytes(e, P(C: UInt32(channels), taps: UInt32(taps), xld: UInt32(x.ld), ldy: UInt32(y.ld), eps: eps), index: 6)
        ce.dispatchThreads(MTLSize(width: rows * 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// y = silu(g[..<hidden]) * g[hidden...] per row.
    func swiglu(_ e: Encoder, _ g: ActView, rows: Int, hidden: Int, _ y: ActView) {
        struct P { var H, ldg, ldy: UInt32 }
        e.start(swiglu)
        let ce = e.ce
        ce.setBuffer(g.buffer, offset: g.offset, index: 0)
        ce.setBuffer(y.buffer, offset: y.offset, index: 1)
        setBytes(e, P(H: UInt32(hidden), ldg: UInt32(g.ld), ldy: UInt32(y.ld)), index: 2)
        ce.dispatchThreads(MTLSize(width: hidden, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Zeroes the first `bytes` (a multiple of 16) of `buffer`.
    func zero(_ e: Encoder, _ buffer: MTLBuffer, bytes: Int) {
        precondition(bytes % 16 == 0 && bytes <= buffer.length)
        e.start(zero)
        e.ce.setBuffer(buffer, offset: 0, index: 0)
        e.ce.dispatchThreads(MTLSize(width: bytes / 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Row m of `codes` (uint32, 16 per row) to [semantic vector | sum of the 15 acoustic vectors].
    func rvqSum(_ e: Encoder, codebooks: GPUTensor, codes: MTLBuffer, codesOffset: Int, rows: Int, _ y: ActView) {
        struct P { var books, dim, vocab, ldy, maxCode: UInt32 }
        let dim = CodecShape.codebookDim
        e.start(rvqSum, weightBytes: CodecShape.codebooks * dim * 4)
        let ce = e.ce
        ce.setBuffer(codebooks.buffer, offset: codebooks.offset, index: 0)
        ce.setBuffer(codes, offset: codesOffset, index: 1)
        ce.setBuffer(y.buffer, offset: y.offset, index: 2)
        setBytes(e, P(books: UInt32(CodecShape.codebooks), dim: UInt32(dim), vocab: UInt32(CodecShape.codebookSize), ldy: UInt32(y.ld), maxCode: UInt32(CodecShape.codebookSize - 1)), index: 3)
        ce.dispatchThreads(MTLSize(width: dim, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// One residual quantizer step per row of x: nearest entry of `codebook` ([2048][256], with
    /// its squared norms) into `codes[row][slot]`, entry subtracted from x.
    func vqNearest(_ e: Encoder, _ x: ActView, rows: Int, codebook: GPUTensor, norms: GPUTensor, codes: MTLBuffer, slot: Int) {
        struct P { var dim, vocab, ldx, books, slot: UInt32 }
        let dim = CodecShape.codebookDim
        e.start(vqNearest, weightBytes: CodecShape.codebookSize * dim * 4)
        let ce = e.ce
        ce.setBuffer(x.buffer, offset: x.offset, index: 0)
        ce.setBuffer(codebook.buffer, offset: codebook.offset, index: 1)
        ce.setBuffer(norms.buffer, offset: norms.offset, index: 2)
        ce.setBuffer(codes, offset: 0, index: 3)
        setBytes(e, P(dim: UInt32(dim), vocab: UInt32(CodecShape.codebookSize), ldx: UInt32(x.ld), books: UInt32(CodecShape.codebooks), slot: UInt32(slot)), index: 4)
        ce.dispatchThreadgroups(MTLSize(width: 1, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// RoPE on rows of [q | k | v] (head dim 64), k and v into ring slot `position % ring` of
    /// caches laid out [head][ring][64]. `table` is `RopeTable`'s buffer.
    func ropeKV(_ e: Encoder, _ qkv: ActView, rows: Int, heads: Int, pos0: Int, ring: Int, k: MTLBuffer, v: MTLBuffer, table: RopeTable) {
        struct P { var M, heads, ldq, pos0, ring: UInt32 }
        precondition(pos0 + rows <= table.positions, "RoPE table holds \(table.positions) positions")
        e.start(ropeKV)
        let ce = e.ce
        ce.setBuffer(qkv.buffer, offset: qkv.offset, index: 0)
        ce.setBuffer(k, offset: 0, index: 1)
        ce.setBuffer(v, offset: 0, index: 2)
        ce.setBuffer(table.buffer, offset: 0, index: 3)
        setBytes(e, P(M: UInt32(rows), heads: UInt32(heads), ldq: UInt32(qkv.ld), pos0: UInt32(pos0), ring: UInt32(ring)), index: 4)
        ce.dispatchThreadgroups(MTLSize(width: 3 * heads, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// Causal attention of `rows` queries (head dim 64, row m at position pos0 + m) over the
    /// `window` positions ending at each query, read from the ring caches.
    func attention(_ e: Encoder, _ q: ActView, rows: Int, heads: Int, pos0: Int, window: Int, ring: Int, k: MTLBuffer, v: MTLBuffer, _ out: ActView) {
        struct P { var M, heads, ldq, ldo, pos0, window, ring: UInt32; var scale: Float }
        let keys = min(window, pos0 + rows)
        precondition(keys <= Self.longWindow && min(window + rows - 1, pos0 + rows) <= ring, "the ring must hold every key the rows see")
        e.start(keys <= Self.shortWindow ? attentionShort : attentionLong)
        let ce = e.ce
        ce.setBuffer(q.buffer, offset: q.offset, index: 0)
        ce.setBuffer(k, offset: 0, index: 1)
        ce.setBuffer(v, offset: 0, index: 2)
        ce.setBuffer(out.buffer, offset: out.offset, index: 3)
        setBytes(e, P(M: UInt32(rows), heads: UInt32(heads), ldq: UInt32(q.ld), ldo: UInt32(out.ld), pos0: UInt32(pos0), window: UInt32(window), ring: UInt32(ring), scale: 0.125), index: 4)
        ce.dispatchThreadgroups(MTLSize(width: heads, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
}

/// cos and sin of position * inverse frequency for head dim 64, theta 10000: per position 32
/// cosines then 32 sines, the angle formed in fp32 as the reference forms it.
struct RopeTable {
    let buffer: MTLBuffer
    let positions: Int

    init(ctx: MetalContext, positions: Int) throws {
        self.positions = positions
        buffer = try ctx.makeBuffer(floats: Self.values(positions: positions), label: "codec.rope")
    }

    static func values(positions: Int) -> [Float] {
        let inv = RopeFrequencies.plain(theta: 10000, dim: 64).values
        var t = [Float](repeating: 0, count: positions * 64)
        for p in 0..<positions {
            for i in 0..<32 {
                let angle = Float(p) * inv[i]
                t[p * 64 + i] = cos(angle)
                t[p * 64 + 32 + i] = sin(angle)
            }
        }
        return t
    }
}

/// Sizes of the Qwen3-TTS 12 Hz tokenizer.
enum CodecShape {
    static let codebooks = 16
    static let codebookSize = 2048
    static let codebookDim = 256
    static let sampleRate = 24000
    static let samplesPerFrame = 1920
    /// Positions a decoder transformer query attends to, itself included.
    static let decoderWindow = 72
}
