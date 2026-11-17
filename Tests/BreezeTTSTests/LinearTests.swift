// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import BreezeTTS

final class LinearTests: GPUTestCase {
    func testShadersCompile() throws {
        try ctx.warmUpPipelines()
    }

    static func geluTanh(_ v: Float) -> Float { 0.5 * v * (1 + tanh(0.7978845608 * (v + 0.044715 * v * v * v))) }
    static func silu(_ v: Float) -> Float { v / (1 + exp(-v)) }

    func linearCase(m: Int, n: Int, k: Int, flags: GemvFlags = [], bias: Bool = false, xScale: Float = 1) throws {
        var rng = RNG(state: UInt64(m * 1000 + n * 7 + k))
        // Rows past m hold NaN: the kernel must never read them.
        let x = rng.floats(m * k, scale: xScale) + [Float](repeating: .nan, count: 8 * k)
        let (w, wf) = try bf16Tensor(rng.floats(n * k, scale: 0.1))
        let (w2, w2f) = try bf16Tensor(rng.floats(n * k, scale: 0.1))
        let (b, bf) = try bf16Tensor(rng.floats(n))
        let y0 = rng.floats(m * n)
        let xb = try f32Buffer(x), yb = try f32Buffer(y0)
        try run { e in
            e.linear(ActView(xb, ld: k), rows: m, Linear(w: w, bias: bias ? b : nil, n: n, k: k), ActView(yb, ld: n), flags: flags, gate: flags.isGated ? w2 : nil)
        }
        var want = [Float](repeating: 0, count: m * n)
        for r in 0..<m {
            let row = x[(r * k)..<((r + 1) * k)]
            for c in 0..<n {
                var v = Self.dot(row, wf[(c * k)..<((c + 1) * k)]) + (bias ? bf[c] : 0)
                if flags.isGated {
                    let up = Self.dot(row, w2f[(c * k)..<((c + 1) * k)])
                    v = (flags.contains(.siluGate) ? Self.silu(v) : Self.geluTanh(v)) * up
                }
                if flags.contains(.gelu) { v = Self.geluTanh(v) }
                if flags.contains(.silu) { v = Self.silu(v) }
                if flags.contains(.accumulate) { v += y0[r * n + c] }
                want[r * n + c] = v
            }
        }
        assertClose(yb.floats(count: m * n), want, atol: 2e-4 * Float(k).squareRoot() / 30, "linear m\(m) n\(n) k\(k) flags \(flags.rawValue) bias \(bias)")
    }

    /// The gemm8 kernel assumes this lane-to-element layout of simdgroup_float8x8.
    func testSimdMatrixLayout() throws {
        let m = try f32Buffer((0..<64).map(Float.init))
        let out = try ctx.makeBuffer(bytes: 64 * 4, label: "out")
        let cb = ctx.queue.makeCommandBuffer()!
        let ce = cb.makeComputeCommandEncoder()!
        ce.setComputePipelineState(try ctx.pipeline("simd_matrix_layout"))
        ce.setBuffer(m, offset: 0, index: 0)
        ce.setBuffer(out, offset: 0, index: 1)
        ce.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        ce.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let v = out.floats(count: 64)
        for lane in 0..<32 {
            let row = ((lane >> 1) & 3) | ((lane >> 4) << 2)
            let col = ((lane & 1) << 1) | (((lane >> 3) & 1) << 2)
            XCTAssertEqual(Int(v[lane * 2]), row * 8 + col, "lane \(lane) element 0")
            XCTAssertEqual(Int(v[lane * 2 + 1]), row * 8 + col + 1, "lane \(lane) element 1")
        }
    }

    /// The decode shapes of the backbone and the depth decoder, one row and the two rows of a
    /// guided step, with their epilogues.
    func testDecodeShapes() throws {
        for m in [1, 2] {
            try linearCase(m: m, n: 4096, k: 2048)
            try linearCase(m: m, n: 2048, k: 2048, flags: .accumulate)
            try linearCase(m: m, n: 6144, k: 2048, flags: .siluGate)
            try linearCase(m: m, n: 2048, k: 6144, flags: .accumulate)
            try linearCase(m: m, n: 1536, k: 1024)
            try linearCase(m: m, n: 8192, k: 1024, flags: .siluGate)
            try linearCase(m: m, n: 1024, k: 8192, flags: .accumulate)
        }
    }

    /// Output counts that are no multiple of anything (the two heads) and K = 1152, which the
    /// scalar kernel walks with a short last stride for half its lanes.
    func testHeadAndEncoderShapes() throws {
        try linearCase(m: 1, n: 2052, k: 2048)
        try linearCase(m: 2, n: 2052, k: 2048)
        try linearCase(m: 2, n: 2051, k: 1024)
        try linearCase(m: 1, n: 2048, k: 1152)
        try linearCase(m: 2, n: 1536, k: 1152)
        try linearCase(m: 1, n: 1152, k: 6912, flags: .accumulate)
        try linearCase(m: 1, n: 6912, k: 1152, flags: .geluGate)
        try linearCase(m: 1, n: 37, k: 64, bias: true)
    }

    /// Every simdgroup count the scalar kernel can be dispatched with gives the same result.
    func testGemvSimdgroupCounts() throws {
        defer { Encoder.gemvSimdgroups = 4 }
        for sg in [1, 2, 8] {
            Encoder.gemvSimdgroups = sg
            try linearCase(m: 2, n: 2051, k: 1024, flags: .accumulate)
            try linearCase(m: 1, n: 1030, k: 1152, flags: .siluGate)
        }
    }

    /// 3 to 15 rows take the matrix-unit kernel: the 4 rows of a guided depth prefill, and its
    /// column-tile and K-split variants with a K tail and N tails.
    func testSmallRowShapes() throws {
        try linearCase(m: 4, n: 1024, k: 2048)
        try linearCase(m: 4, n: 1536, k: 1024)
        try linearCase(m: 4, n: 8192, k: 1024, flags: .siluGate)
        try linearCase(m: 4, n: 1024, k: 8192, flags: .accumulate)
        try linearCase(m: 9, n: 2051, k: 1152)
        try linearCase(m: 5, n: 6912, k: 1152, flags: .geluGate)
        defer { Encoder.columnTilesOverride = nil; Encoder.kSplitOverride = nil }
        for nc in [1, 2, 4] {
            for ks in [4, 8] {
                Encoder.columnTilesOverride = nc
                Encoder.kSplitOverride = ks
                try linearCase(m: 6, n: 1024, k: 1024, flags: .gelu, bias: true)
                try linearCase(m: 4, n: 896, k: 1088, flags: .siluGate)
                try linearCase(m: 3, n: 1152, k: 1152, flags: .accumulate)
            }
        }
    }

    /// Prefill shapes route through the tiled GEMM (16 rows and up, N % 64 == 0, K % 32 == 0),
    /// K = 1152 included; a head-shaped N falls back to the matrix-unit kernel.
    func testPrefillShapes() throws {
        XCTAssertTrue(Kernels.usesGemm(rows: 40, n: 4096, k: 2048))
        XCTAssertTrue(Kernels.usesGemm(rows: 40, n: 1536, k: 1152))
        XCTAssertFalse(Kernels.usesGemm(rows: 40, n: 2052, k: 2048))
        XCTAssertFalse(Kernels.usesGemm(rows: 8, n: 4096, k: 2048))
        try linearCase(m: 40, n: 4096, k: 2048)
        try linearCase(m: 33, n: 2048, k: 6144, flags: .accumulate)
        try linearCase(m: 17, n: 6144, k: 2048, flags: .siluGate)
        try linearCase(m: 40, n: 1536, k: 1152)
        try linearCase(m: 21, n: 6912, k: 1152, flags: .geluGate)
        try linearCase(m: 65, n: 1152, k: 6912, flags: .accumulate)
        try linearCase(m: 19, n: 2052, k: 2048)
        try linearCase(m: 16, n: 64, k: 32, flags: .gelu, bias: true)
    }

    /// Pre-activations of tens are normal inside an MLP; tanh must not overflow there.
    func testLargeActivations() throws {
        try linearCase(m: 6, n: 256, k: 1024, flags: .gelu, bias: true, xScale: 8)
        try linearCase(m: 1, n: 256, k: 1152, flags: .geluGate, xScale: 8)
        try linearCase(m: 1, n: 256, k: 1024, flags: .siluGate, xScale: 8)
    }

    func normCase(rows: Int, k: Int, ld: Int, eps: Float, onePlus: Bool, accumulate: Bool) throws {
        var rng = RNG(state: UInt64(rows * 131 + k))
        let x = rng.floats(rows * ld, scale: 3)
        let (w, wf) = try bf16Tensor(rng.floats(k))
        let y0 = rng.floats(rows * ld)
        let xb = try f32Buffer(x), yb = try f32Buffer(y0)
        try run { e in e.rmsnorm(ActView(xb, ld: ld), rows: rows, width: k, weight: w, ActView(yb, ld: ld), eps: eps, onePlusWeight: onePlus, accumulate: accumulate) }
        var want = y0
        for r in 0..<rows {
            let row = x[(r * ld)..<(r * ld + k)]
            let s = 1 / (Self.dot(row, row) / Float(k) + eps).squareRoot()
            for i in 0..<k {
                let v = x[r * ld + i] * s * (onePlus ? 1 + wf[i] : wf[i])
                want[r * ld + i] = accumulate ? y0[r * ld + i] + v : v
            }
        }
        assertClose(yb.floats(count: rows * ld), want, atol: 1e-5, "rmsnorm rows \(rows) k \(k) onePlus \(onePlus) accumulate \(accumulate)")
    }

    /// Both conventions, full width with a row stride wider than the norm (untouched columns keep
    /// their values), and per head: one row per head, width the head dim.
    func testRMSNorm() throws {
        try normCase(rows: 5, k: 2048, ld: 2048, eps: 1e-6, onePlus: false, accumulate: false)
        try normCase(rows: 2, k: 1024, ld: 2048, eps: 1e-5, onePlus: false, accumulate: false)
        try normCase(rows: 3, k: 1152, ld: 1152, eps: 1e-6, onePlus: true, accumulate: false)
        try normCase(rows: 3, k: 1152, ld: 1152, eps: 1e-6, onePlus: true, accumulate: true)
        try normCase(rows: 24, k: 128, ld: 128, eps: 1e-6, onePlus: false, accumulate: false)
        try normCase(rows: 10, k: 256, ld: 256, eps: 1e-6, onePlus: true, accumulate: false)
    }
}
