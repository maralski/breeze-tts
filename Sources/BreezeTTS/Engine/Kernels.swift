// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// A bf16 (or fp32) tensor living in a GPU buffer at a byte offset.
struct GPUTensor {
    let buffer: MTLBuffer
    let offset: Int
    let count: Int
}

/// y = x W^T (+ bias). `w` is bf16 [n, k] row-major.
struct Linear {
    let w: GPUTensor
    let bias: GPUTensor?
    let n: Int
    let k: Int
}

struct GemvFlags: OptionSet {
    let rawValue: UInt32
    static let bias = GemvFlags(rawValue: 1)
    /// Adds the result to y: the residual add of an output projection.
    static let accumulate = GemvFlags(rawValue: 2)
    static let gelu = GemvFlags(rawValue: 4)
    /// silu(x W^T) * (x W2^T), SwiGLU's gate and up in one read of x.
    static let siluGate = GemvFlags(rawValue: 8)
    static let silu = GemvFlags(rawValue: 16)
    /// gelu_tanh(x W^T) * (x W2^T), Gemma's gated MLP.
    static let geluGate = GemvFlags(rawValue: 32)
    var isGated: Bool { !isDisjoint(with: [.siluGate, .geluGate]) }
}

/// A projection as the scalar kernel sees it: rows of x, output columns, K.
struct GemvShape: Hashable {
    let m, n, k: Int
}

/// How the scalar projection kernel covers a shape: simdgroups per threadgroup, and output
/// columns per simdgroup (one load of x feeds that many weight rows). Neither changes a sum, so
/// every layout gives the same bits.
struct GemvLayout: Hashable {
    var simdgroups: Int
    var columns: Int
    static let simdgroupChoices = [1, 2, 4, 8, 16, 32]
    /// MingTTS's layout, the one this engine started with.
    static let fourSimdgroups = GemvLayout(simdgroups: 4, columns: 1)
    /// One simdgroup per threadgroup computing two output columns, for every shape. Measured on
    /// an M3 Max over guided frames, schedules interleaved in one process. In the first minute
    /// after a rest it costs 0.4 to 1 ms of a 43 to 46 ms frame against `fourSimdgroups`. Once
    /// generation is sustained it is ahead, by 1.0 to 6.6 ms per frame of the token loop in four
    /// 150-frame runs with nothing else running (--bench-codec --token-loop with
    /// BREEZE_CODEC_SCHEDULES=sg4), and by 6 to 10 ms while other processes used the GPU and the
    /// CPUs. No per-shape table did better there. Sustained generation is the engine's job, so
    /// this layout ships.
    static let shipped = GemvLayout(simdgroups: 1, columns: 2)

    init(simdgroups: Int, columns: Int) {
        self.simdgroups = simdgroups
        self.columns = columns
    }

    /// "sgN" or "sgNrC".
    init?(_ spec: String) {
        let parts = spec.dropFirst(2).split(separator: "r").compactMap { Int($0) }
        guard spec.hasPrefix("sg"), let sg = parts.first, Self.simdgroupChoices.contains(sg), parts.count <= 2 else { return nil }
        let columns = parts.count == 2 ? parts[1] : 1
        guard Kernels.gemvColumns.contains(columns) else { return nil }
        self.init(simdgroups: sg, columns: columns)
    }
}

/// An fp32 activation matrix view: rows of floats at `offset` bytes, row stride `ld` floats.
struct ActView {
    let buffer: MTLBuffer
    let offset: Int
    let ld: Int

    init(_ buffer: MTLBuffer, offset: Int = 0, ld: Int) {
        self.buffer = buffer
        self.offset = offset
        self.ld = ld
    }

    func row(_ r: Int) -> ActView { ActView(buffer, offset: offset + r * ld * 4, ld: ld) }
    func column(_ c: Int) -> ActView { ActView(buffer, offset: offset + c * 4, ld: ld) }
    /// The same rows read with another stride, to pick every nth row for instance.
    func strided(_ ld: Int) -> ActView { ActView(buffer, offset: offset, ld: ld) }
}

/// The rows of one dispatch: `sequences` independent sequences of `rows` new rows each, laid
/// out sequence-major, sequence s starting at position `positions[s]`. A guided decode step is
/// two sequences of one row at different positions; prefill is one sequence of many rows.
struct RowBlock: Equatable {
    static let maxSequences = 4
    let sequences: Int
    let rows: Int
    let positions: [Int]

    init(rows: Int, positions: [Int]) {
        precondition(!positions.isEmpty && positions.count <= Self.maxSequences && rows > 0)
        sequences = positions.count
        self.rows = rows
        self.positions = positions
    }

    var count: Int { sequences * rows }
    var packedPositions: (UInt32, UInt32, UInt32, UInt32) {
        let p = positions.map(UInt32.init) + [UInt32](repeating: 0, count: Self.maxSequences - sequences)
        return (p[0], p[1], p[2], p[3])
    }
}

/// One layer's K and V caches: fp32 [sequence][kvHead][capacity][headDim] each.
struct KVCacheLayer {
    let k: GPUTensor
    let v: GPUTensor
    let capacity: Int
    /// Floats in one sequence's block.
    let seqStride: Int

    /// The cache as seen from sequence `first` on.
    func sequences(from first: Int) -> KVCacheLayer {
        KVCacheLayer(k: GPUTensor(buffer: k.buffer, offset: k.offset + first * seqStride * 4, count: k.count),
                     v: GPUTensor(buffer: v.buffer, offset: v.offset + first * seqStride * 4, count: v.count),
                     capacity: capacity, seqStride: seqStride)
    }
}

/// Per-head RMSNorm applied to q and k before RoPE; the tensor is q_norm then k_norm.
enum QKNorm {
    case none
    case weight(GPUTensor, eps: Float)
    case onePlusWeight(GPUTensor, eps: Float)
}

enum SamplerKind { case backbone, depth }

/// Attention size limits the kernels are compiled with.
enum AttentionLimits {
    static let shortKeys = 17
}

/// Counts what one encoded stretch asks of the GPU.
final class DispatchTally {
    private(set) var dispatches = 0
    /// Bytes of weights the dispatches stream, embeddings and norm weights included.
    private(set) var weightBytes = 0
    func add(weightBytes: Int) {
        dispatches += 1
        self.weightBytes += weightBytes
    }
}

/// The compiled pipelines, resolved once.
final class Kernels: @unchecked Sendable {
    let ctx: MetalContext
    /// Scalar GEMV variants by (rows of x per simdgroup, output columns per simdgroup).
    struct GV: Hashable { let mr, nr: Int }
    static let gemvColumns = [1, 2]
    let gemv: [GV: MTLComputePipelineState]
    /// gemm8 variants by (K split, column tiles).
    struct G8: Hashable { let ks, nc: Int }
    static let gemm8Variants = [G8(ks: 4, nc: 1), G8(ks: 4, nc: 2), G8(ks: 4, nc: 4), G8(ks: 8, nc: 1), G8(ks: 8, nc: 2), G8(ks: 8, nc: 4), G8(ks: 16, nc: 1), G8(ks: 16, nc: 2)]
    let gemm8: [G8: MTLComputePipelineState]
    let gemm: MTLComputePipelineState
    let rmsnorm, copyRows, normRopeKV, attentionShort, attentionWindow, embedCodes, sampleBackbone, sampleDepth: MTLComputePipelineState
    /// Backbone attention pipelines by simdgroups per (row, KV head) item.
    static let causalSplits = [1, 4, 8]
    let attentionCausal: [Int: MTLComputePipelineState]

    init(ctx: MetalContext) throws {
        self.ctx = ctx
        var g: [GV: MTLComputePipelineState] = [:]
        for mr in [1, 2] {
            for nr in Kernels.gemvColumns { g[GV(mr: mr, nr: nr)] = try ctx.pipeline("gemv_\(mr)_\(nr)") }
        }
        gemv = g
        var g8: [G8: MTLComputePipelineState] = [:]
        for v in Kernels.gemm8Variants { g8[v] = try ctx.pipeline("gemm8_\(v.ks)_\(v.nc)") }
        gemm8 = g8
        gemm = try ctx.pipeline("gemm_32x64x32")
        rmsnorm = try ctx.pipeline("rmsnorm")
        copyRows = try ctx.pipeline("copy_rows")
        normRopeKV = try ctx.pipeline("norm_rope_kv")
        var causal: [Int: MTLComputePipelineState] = [:]
        for s in Kernels.causalSplits { causal[s] = try ctx.pipeline("attention_causal_\(s)") }
        attentionCausal = causal
        attentionShort = try ctx.pipeline("attention_short")
        attentionWindow = try ctx.pipeline("attention_window")
        embedCodes = try ctx.pipeline("embed_codes")
        sampleBackbone = try ctx.pipeline("sample_backbone")
        sampleDepth = try ctx.pipeline("sample_depth")
    }

    /// The tiled GEMM takes over at this many rows (prefill), when the shape tiles. Below it
    /// gemm8's 8-row chunks give the GPU more threadgroups: measured on an M3 Max over prompt
    /// encoding plus prefill (`--bench-prompt` with BREEZE_GEMM_CUTS), gemm8 takes 66 ms
    /// against 81 at 96 rows and 130 against 138 at 192; at 256 the two tie, and from 320 rows
    /// the GEMM wins (576 against 676 ms at 768). Overridable for the bench and the parity test
    /// of the GEMM path.
    nonisolated(unsafe) static var gemmMinRows = 256
    /// Prefill blocks at least this tall give each (row, KV head) of the causal attention one simdgroup.
    static let prefillAttentionRows = 16
    static let gemmBM = 32, gemmBN = 64, gemmBK = 32
    static func usesGemm(rows m: Int, n: Int, k: Int) -> Bool {
        m >= gemmMinRows && n % gemmBN == 0 && k % gemmBK == 0
    }
}

/// Encodes kernels onto one compute encoder. Every method appends one dispatch. On a concurrent
/// encoder (`fenced`) each dispatch is preceded by a buffer barrier, except the later members of
/// a `parallel` group, which the caller promises touch disjoint buffers.
final class Encoder {
    let k: Kernels
    /// The pass the next dispatch goes to. A traced encoder replaces it at every dispatch.
    private(set) var ce: MTLComputeCommandEncoder
    let fenced: Bool
    let tally: DispatchTally?
    private let trace: DispatchTrace?
    private var inGroup = false
    private var groupStarted = false

    init(k: Kernels, ce: MTLComputeCommandEncoder, fenced: Bool = false, tally: DispatchTally? = nil) {
        self.k = k
        self.ce = ce
        self.fenced = fenced
        self.tally = tally
        trace = nil
    }

    /// Every dispatch in a compute pass of its own, timed by `trace`.
    init(k: Kernels, trace: DispatchTrace) {
        self.k = k
        ce = trace.idlePass()
        fenced = false
        tally = nil
        self.trace = trace
    }

    /// Dispatches encoded in `body` may run at the same time.
    func parallel(_ body: () -> Void) {
        inGroup = true
        groupStarted = false
        body()
        inGroup = false
    }

    private func begin(_ pipeline: MTLComputePipelineState, weightBytes: Int = 0) {
        tally?.add(weightBytes: weightBytes)
        if let trace {
            ce.endEncoding()
            ce = trace.pass(for: pipeline, weightBytes: weightBytes)
        }
        if fenced && !(inGroup && groupStarted) { ce.memoryBarrier(scope: .buffers) }
        if inGroup { groupStarted = true }
        ce.setComputePipelineState(pipeline)
    }

    /// `begin` for kernels encoded from other files (the codec's).
    func start(_ pipeline: MTLComputePipelineState, weightBytes: Int = 0) { begin(pipeline, weightBytes: weightBytes) }

    private func setBytes<T>(_ value: T, index: Int) {
        withUnsafeBytes(of: value) { ce.setBytes($0.baseAddress!, length: $0.count, index: index) }
    }

    private func bind(_ t: GPUTensor, _ index: Int) { ce.setBuffer(t.buffer, offset: t.offset, index: index) }
    private func bind(_ v: ActView, _ index: Int) { ce.setBuffer(v.buffer, offset: v.offset, index: index) }

    /// Simdgroups splitting K in one gemm8 threadgroup (MingTTS rule: eight keeps a decode shape's
    /// threadgroups in one wave). Overridable for the micro-bench.
    nonisolated(unsafe) static var kSplitOverride: Int?
    static func kSplit(forK k: Int) -> Int {
        if let s = kSplitOverride { return s }
        for ks in [8, 4] where k % (8 * ks) == 0 { return ks }
        preconditionFailure("gemm8 needs K % 32 == 0, got \(k)")
    }

    /// Column tiles per gemm8 simdgroup (MingTTS's measured rule: two above K 1024). The kernel
    /// needs N % (8 nc) == 0 above one tile. Overridable for the micro-bench.
    nonisolated(unsafe) static var columnTilesOverride: Int?
    static func columnTiles(forN n: Int, k: Int) -> Int {
        let want = columnTilesOverride ?? (k > 1024 ? 2 : 1)
        for nc in [want, 2, 1] where nc <= 4 && n % (8 * nc) == 0 { return nc }
        return 1
    }

    /// Row counts up to this take the scalar kernel. Overridable for the micro-bench.
    nonisolated(unsafe) static var scalarRowsMax = 2
    /// Forces one scalar-kernel layout for every shape: the micro-bench's and the tests' knob.
    nonisolated(unsafe) static var gemvLayoutOverride: GemvLayout?
    /// Layouts for single shapes, ahead of the rule: the frame bench's knob.
    nonisolated(unsafe) static var gemvShapeOverrides: [GemvShape: GemvLayout] = [:]
    static func gemvLayout(_ shape: GemvShape) -> GemvLayout {
        gemvLayoutOverride ?? gemvShapeOverrides[shape] ?? .shipped
    }
    /// Simdgroups sharing one (row, KV head) of the backbone attention at decode. Measured at 300
    /// cached keys on an M3 Max: 81 us per dispatch with one, 34 with four, 22 with eight.
    /// Overridable for the micro-bench.
    nonisolated(unsafe) static var causalDecodeSplit = 8

    /// y[m, n] = sum_k x[m, k] w[n, k] with the epilogue in `flags`. `gate` holds the second
    /// matrix for a gated flag, in which case `linear` is the gate projection. Rows of x past m
    /// are never read. One or two rows take the scalar kernel (any N, K % 8 == 0), prefill shapes
    /// that tile (16 rows and up, N % 64 == 0, K % 32 == 0) the tiled GEMM, the rest the
    /// matrix-unit kernel (any N, K % 32 == 0).
    func linear(_ x: ActView, rows m: Int, _ linear: Linear, _ y: ActView, flags: GemvFlags = [], gate: GPUTensor? = nil) {
        struct P { var M, N, K, ldx, ldy, flags, sg: UInt32 }
        precondition(flags.isGated == (gate != nil), "a gated epilogue needs the second weight")
        var f = flags
        if linear.bias != nil { f.insert(.bias) }
        let weights = linear.n * linear.k * 2
        let groups: MTLSize, threads: Int
        var sg = 0
        if Kernels.usesGemm(rows: m, n: linear.n, k: linear.k) {
            begin(k.gemm, weightBytes: weights * (gate == nil ? 1 : 2))
            groups = MTLSize(width: linear.n / Kernels.gemmBN, height: (m + Kernels.gemmBM - 1) / Kernels.gemmBM, depth: 1)
            threads = 128
        } else if m <= Encoder.scalarRowsMax {
            precondition(linear.k % 8 == 0, "gemv needs K % 8 == 0, got \(linear.k)")
            let layout = Encoder.gemvLayout(GemvShape(m: m, n: linear.n, k: linear.k))
            sg = layout.simdgroups
            let nr = layout.columns
            begin(k.gemv[Kernels.GV(mr: m, nr: nr)]!, weightBytes: weights * (gate == nil ? 1 : 2))
            groups = MTLSize(width: (linear.n + sg * nr - 1) / (sg * nr), height: 1, depth: 1)
            threads = 32 * sg
        } else {
            let ks = Encoder.kSplit(forK: linear.k), nc = Encoder.columnTiles(forN: linear.n, k: linear.k)
            guard let pipeline = k.gemm8[Kernels.G8(ks: ks, nc: nc)] else { preconditionFailure("gemm8 has no (ks \(ks), nc \(nc)) variant") }
            begin(pipeline, weightBytes: weights * (gate == nil ? 1 : 2))
            groups = MTLSize(width: (linear.n + 8 * nc - 1) / (8 * nc), height: (m + 7) / 8, depth: 1)
            threads = 32 * ks
        }
        bind(x, 0)
        bind(linear.w, 1)
        bind(gate ?? linear.w, 2)
        bind(linear.bias ?? linear.w, 3)
        bind(y, 4)
        setBytes(P(M: UInt32(m), N: UInt32(linear.n), K: UInt32(linear.k), ldx: UInt32(x.ld), ldy: UInt32(y.ld), flags: f.rawValue, sg: UInt32(sg)), index: 5)
        ce.dispatchThreadgroups(groups, threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    /// y = x * rsqrt(mean(x^2) + eps) * w per row, or * (1 + w) with `onePlusWeight` (Gemma);
    /// `accumulate` adds the result to y. Width is a multiple of 4. A per-head norm is this with
    /// one row per head and `x.ld` = the head dim.
    func rmsnorm(_ x: ActView, rows: Int, width: Int, weight: GPUTensor, _ y: ActView, eps: Float, onePlusWeight: Bool = false, accumulate: Bool = false) {
        struct P { var K, ldx, ldy: UInt32; var eps: Float; var flags: UInt32 }
        precondition(width % 4 == 0)
        begin(k.rmsnorm, weightBytes: width * 2)
        bind(x, 0)
        bind(weight, 1)
        bind(y, 2)
        setBytes(P(K: UInt32(width), ldx: UInt32(x.ld), ldy: UInt32(y.ld), eps: eps, flags: (onePlusWeight ? 1 : 0) | (accumulate ? 2 : 0)), index: 3)
        ce.dispatchThreads(MTLSize(width: rows * 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func copy(_ x: ActView, count: Int, to y: ActView) {
        begin(k.copyRows)
        bind(x, 0)
        bind(y, 1)
        ce.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(count, 256), height: 1, depth: 1))
    }

    /// Finishes rows of [q | k | v] projections: optional per-head norm of q and k, rotate-half
    /// RoPE from `invFreq` (headDim / 2 floats), q in place, k and v into the cache at each row's
    /// position.
    func normRopeKV(_ qkv: ActView, block: RowBlock, heads hq: Int, kvHeads hkv: Int, headDim: Int, cache: KVCacheLayer, invFreq: MTLBuffer, norm: QKNorm) {
        struct P { var B, M, hq, hkv, D, ldq, cap, seqStride, normMode: UInt32; var eps: Float; var pos0: (UInt32, UInt32, UInt32, UInt32) }
        precondition(headDim % 64 == 0 && headDim <= 256 && invFreq.length >= headDim * 2)
        precondition(block.positions.max()! + block.rows <= cache.capacity)
        var mode: UInt32 = 0, eps: Float = 0
        var weight: GPUTensor?
        switch norm {
        case .none: break
        case .weight(let w, let e): (mode, eps, weight) = (1, e, w)
        case .onePlusWeight(let w, let e): (mode, eps, weight) = (2, e, w)
        }
        begin(k.normRopeKV, weightBytes: weight == nil ? 0 : headDim * 4)
        bind(qkv, 0)
        bind(cache.k, 1)
        bind(cache.v, 2)
        ce.setBuffer(invFreq, offset: 0, index: 3)
        bind(weight ?? cache.k, 4)
        setBytes(P(B: UInt32(block.sequences), M: UInt32(block.rows), hq: UInt32(hq), hkv: UInt32(hkv), D: UInt32(headDim), ldq: UInt32(qkv.ld),
                   cap: UInt32(cache.capacity), seqStride: UInt32(cache.seqStride), normMode: mode, eps: eps, pos0: block.packedPositions), index: 5)
        ce.dispatchThreadgroups(MTLSize(width: hq + 2 * hkv, height: block.count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// (a) Causal attention of the block's rows over the cache, head dim 128, two query heads per
    /// KV head. Any number of cached keys. Few rows spread each (row, KV head) over several
    /// simdgroups; prefill has items enough and gives each one.
    func attentionCausal(_ q: ActView, block: RowBlock, heads hq: Int, kvHeads hkv: Int, cache: KVCacheLayer, _ out: ActView) {
        struct P { var B, M, hq, hkv, ldq, ldo, cap, seqStride: UInt32; var scale: Float; var pos0: (UInt32, UInt32, UInt32, UInt32) }
        precondition(hq == 2 * hkv, "attention_causal is compiled for two query heads per KV head")
        let split = block.rows < Kernels.prefillAttentionRows ? Encoder.causalDecodeSplit : 1
        begin(k.attentionCausal[split]!)
        bind(q, 0)
        bind(cache.k, 1)
        bind(cache.v, 2)
        bind(out, 3)
        setBytes(P(B: UInt32(block.sequences), M: UInt32(block.rows), hq: UInt32(hq), hkv: UInt32(hkv), ldq: UInt32(q.ld), ldo: UInt32(out.ld),
                   cap: UInt32(cache.capacity), seqStride: UInt32(cache.seqStride), scale: 1 / Float(128).squareRoot(), pos0: block.packedPositions), index: 4)
        ce.dispatchThreadgroups(MTLSize(width: hkv, height: block.count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32 * split, height: 1, depth: 1))
    }

    /// (b) The depth decoder's attention, head dim 128: RoPE on the new rows' q, k and v into the
    /// cache, and causal attention over at most `AttentionLimits.shortKeys` keys, in one dispatch.
    /// Every sequence is at position `pos0`.
    func attentionShort(_ qkv: ActView, sequences: Int, rows: Int, pos0: Int, heads hq: Int, kvHeads hkv: Int, cache: KVCacheLayer, invFreq: MTLBuffer, _ out: ActView) {
        struct P { var B, M, hq, hkv, ldq, ldo, cap, seqStride: UInt32; var scale: Float; var pos0: UInt32 }
        precondition(pos0 + rows <= min(cache.capacity, AttentionLimits.shortKeys) && hq % hkv == 0 && invFreq.length >= 64 * 4)
        begin(k.attentionShort)
        bind(qkv, 0)
        bind(cache.k, 1)
        bind(cache.v, 2)
        ce.setBuffer(invFreq, offset: 0, index: 3)
        bind(out, 4)
        setBytes(P(B: UInt32(sequences), M: UInt32(rows), hq: UInt32(hq), hkv: UInt32(hkv), ldq: UInt32(qkv.ld), ldo: UInt32(out.ld),
                   cap: UInt32(cache.capacity), seqStride: UInt32(cache.seqStride), scale: 1 / Float(128).squareRoot(), pos0: UInt32(pos0)), index: 5)
        ce.dispatchThreadgroups(MTLSize(width: (sequences * rows * hq + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    /// (c) The text encoder's attention: `rows` queries at positions row0... of one sequence of
    /// `total` keys already in `keys` and `values` (fp32 [total, 256]), four query heads on one KV
    /// head, each query seeing `before` keys behind it and `after` ahead. `window: nil` is full
    /// bidirectional attention.
    func attentionWindow(_ q: ActView, rows: Int, row0: Int = 0, total: Int, window: (before: Int, after: Int)?, keys: GPUTensor, values: GPUTensor, scale: Float, _ out: ActView) {
        struct P { var M, row0, total, before, after, ldq, ldo: UInt32; var scale: Float }
        begin(k.attentionWindow)
        bind(q, 0)
        bind(keys, 1)
        bind(values, 2)
        bind(out, 3)
        setBytes(P(M: UInt32(rows), row0: UInt32(row0), total: UInt32(total), before: UInt32(window?.before ?? total), after: UInt32(window?.after ?? total),
                   ldq: UInt32(q.ld), ldo: UInt32(out.ld), scale: scale), index: 4)
        ce.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }

    /// out[r] = sum over c < count of table[codes[r * rowStride + firstCodebook + c] + (firstCodebook + c) * vocab],
    /// added in ascending c. `codes` is uint32 at `codesOffset` bytes. rowStride 0 gives every
    /// output row the same frame.
    func embedCodes(table: GPUTensor, width: Int, vocab: Int, codes: MTLBuffer, codesOffset: Int, firstCodebook: Int, count: Int, rows: Int, rowStride: Int = 0, _ out: ActView) {
        struct P { var width, count, vocab, ldy, q0, rowStride: UInt32 }
        precondition(width % 4 == 0)
        begin(k.embedCodes, weightBytes: count * width * 2)
        bind(table, 0)
        ce.setBuffer(codes, offset: codesOffset, index: 1)
        bind(out, 2)
        setBytes(P(width: UInt32(width), count: UInt32(count), vocab: UInt32(vocab), ldy: UInt32(out.ld), q0: UInt32(firstCodebook), rowStride: UInt32(rowStride)), index: 3)
        ce.dispatchThreads(MTLSize(width: width / 4, height: rows, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// Draws one token from `logits` (one row, or [conditional, unconditional] with `guidance`)
    /// by the rule in Sampling.swift, into `codes[slot]` using `uniforms[slot]`. The backbone
    /// kind reads and extends the `seen` set; both set bit 0 of `status` on unusable logits.
    func sample(_ kind: SamplerKind, logits: ActView, guidance: Float?, settings: SamplerSettings, uniforms: MTLBuffer, uniformsOffset: Int,
                codes: MTLBuffer, codesOffset: Int, slot: Int, seen: MTLBuffer, status: MTLBuffer) {
        struct P { var N, rows, ld, topK, greedy, slot, suppressLo, suppressHi, clampMax: UInt32; var cfgScale, temperature, penalty: Float }
        let n = kind == .backbone ? BreezeShape.backboneVocab : BreezeShape.codebookVocab
        begin(kind == .backbone ? k.sampleBackbone : k.sampleDepth)
        bind(logits, 0)
        ce.setBuffer(uniforms, offset: uniformsOffset, index: 1)
        ce.setBuffer(codes, offset: codesOffset, index: 2)
        ce.setBuffer(seen, offset: 0, index: 3)
        ce.setBuffer(status, offset: 0, index: 4)
        setBytes(P(N: UInt32(n), rows: guidance == nil ? 1 : 2, ld: UInt32(logits.ld), topK: UInt32(settings.topK), greedy: settings.greedy ? 1 : 0, slot: UInt32(slot),
                   suppressLo: UInt32(BreezeShape.reservedIDs.lowerBound), suppressHi: UInt32(BreezeShape.reservedIDs.upperBound), clampMax: UInt32(BreezeShape.codebookVocab - 1),
                   cfgScale: guidance ?? 1, temperature: settings.temperature, penalty: settings.repetitionPenalty), index: 5)
        ce.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}
