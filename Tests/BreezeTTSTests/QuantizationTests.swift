// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
import zlib
@testable import BreezeTTS

/// The int8 depth-decoder path: the load-time quantizer, the int8 projection kernel against a
/// double-precision evaluation of the same quantized weights, and the whole frame against the
/// dequantized reference weights on the bf16 kernels.
final class QuantizationTests: GPUTestCase {
    struct Quantized {
        let q: [Int8]
        let scales: [Float16]
        /// q * scale per weight, exact.
        var values: [Float] { q.indices.map { Int8Stripes.weight(q[$0], scales[($0 / k) * Int8Stripes.lanes + ($0 % k / Int8Stripes.run) % Int8Stripes.lanes]) } }
        let k: Int
    }

    static func quantize(_ bits: [UInt16], n: Int, k: Int) -> Quantized {
        var q = [Int8](repeating: 0, count: n * k)
        var scales = [Float16](repeating: 0, count: n * Int8Stripes.lanes)
        bits.withUnsafeBufferPointer { Int8Stripes.quantize($0.baseAddress!, n: n, k: k, q: &q, scales: &scales) }
        return Quantized(q: q, scales: scales, k: k)
    }

    func int8Linear(_ z: Quantized, n: Int, k: Int) throws -> Linear {
        let w = try ctx.makeBuffer(bytes: n * k, label: "q")
        _ = z.q.withUnsafeBytes { memcpy(w.contents(), $0.baseAddress!, $0.count) }
        let s = try ctx.makeBuffer(bytes: z.scales.count * 2, label: "scales")
        _ = z.scales.withUnsafeBytes { memcpy(s.contents(), $0.baseAddress!, $0.count) }
        return Linear(w: GPUTensor(buffer: w, offset: 0, count: n * k), bias: nil, n: n, k: k, scales: GPUTensor(buffer: s, offset: 0, count: z.scales.count))
    }

    // MARK: quantizer

    /// Every weight lands within half a step of its stripe's scale, the largest weight of a
    /// stripe maps to 127 in magnitude, and no code is -128.
    func testQuantizerErrorIsHalfAStep() {
        var rng = RNG(state: 11)
        let n = 5, k = 1024
        let bits = rng.floats(n * k, scale: 0.07).map(Self.bf16Bits)
        let z = Self.quantize(bits, n: n, k: k)
        XCTAssertFalse(z.q.contains(-128))
        let back = z.values
        for r in 0..<n {
            for lane in 0..<Int8Stripes.lanes {
                let s = Float(z.scales[r * Int8Stripes.lanes + lane])
                var top: Int8 = 0
                for i in stride(from: lane * Int8Stripes.run, to: k, by: Int8Stripes.period) {
                    for j in i..<(i + Int8Stripes.run) {
                        let w = Self.bf16Round(Float(bitPattern: UInt32(bits[r * k + j]) << 16))
                        XCTAssertLessThanOrEqual(abs(back[r * k + j] - w), 0.5001 * s, "row \(r) k \(j)")
                        top = max(top, Int8(abs(Int(z.q[r * k + j]))))
                    }
                }
                XCTAssertGreaterThanOrEqual(top, 126, "row \(r) lane \(lane) uses the range")
            }
        }
    }

    /// A stripe of zeros quantizes to zeros with a zero scale, not to NaN.
    func testZeroStripe() {
        var bits = [UInt16](repeating: Self.bf16Bits(0.25), count: 512)
        for j in 16..<32 { bits[j] = 0 }
        let z = Self.quantize(bits, n: 1, k: 512)
        XCTAssertEqual(z.scales[1], 0)
        XCTAssertTrue(z.q[16..<32].allSatisfy { $0 == 0 })
        XCTAssertEqual(z.q[0], 127)
        XCTAssertTrue(z.values.allSatisfy { $0.isFinite })
    }

    /// The reference form is the quantized weight rounded to bf16.
    func testReferenceIsTheQuantizedWeightInBF16() {
        var rng = RNG(state: 12)
        let n = 3, k = 1024
        let bits = rng.floats(n * k, scale: 0.05).map(Self.bf16Bits)
        var out = [UInt16](repeating: 0, count: n * k)
        bits.withUnsafeBufferPointer { Int8Stripes.reference($0.baseAddress!, n: n, k: k, into: &out) }
        XCTAssertEqual(out, Self.quantize(bits, n: n, k: k).values.map(Self.bf16Bits))
    }

    /// The Swift quantizer gives the bytes of the Python one the scheme sweep and the study
    /// description use: `tools/quant/depth_quant.py checksum --block 0 --scale free --stripes`.
    func testQuantizerMatchesThePythonSweep() throws {
        guard let weights = TestEnvironment.weightsDirectory else { throw XCTSkip("no Breeze-TTS-2 checkpoint") }
        let store = try ShardedCheckpoint(directory: weights)
        let want: [(String, UInt, UInt)] = [
            ("depth_decoder.model.layers.0.mlp.gate_proj.weight", 0x587b_da95, 0xc8e8_d7f7),
            ("depth_decoder.model.layers.11.mlp.down_proj.weight", 0x8ae1_c760, 0x7608_034b),
            ("depth_decoder.model.layers.5.self_attn.o_proj.weight", 0xd3a1_6486, 0x7571_2ad1),
        ]
        for (name, qSum, scaleSum) in want {
            let shape = try store.entry(name).shape
            let z = try store.withBytes(name) { raw, _ in
                Self.quantize(Array(raw.bindMemory(to: UInt16.self)), n: shape[0], k: shape[1])
            }
            let q = z.q.withUnsafeBytes { crc32(0, $0.baseAddress!.assumingMemoryBound(to: UInt8.self), UInt32($0.count)) }
            let s = z.scales.map { Float($0) }.withUnsafeBytes { crc32(0, $0.baseAddress!.assumingMemoryBound(to: UInt8.self), UInt32($0.count)) }
            XCTAssertEqual(q, qSum, "\(name) codes")
            XCTAssertEqual(s, scaleSum, "\(name) scales")
        }
    }

    // MARK: kernel

    /// One int8 projection against sum_k x[k] * q[n, k] * scale[n, lane(k)] in double precision.
    @discardableResult
    func int8Case(m: Int, n: Int, k: Int, flags: GemvFlags = []) throws -> [Float] {
        var rng = RNG(state: UInt64(m * 1000 + n * 7 + k + 99))
        // Rows past m hold NaN: the kernel must never read them.
        let x = rng.floats(m * k) + [Float](repeating: .nan, count: 8 * k)
        // Stripe scales spread over a factor of 16, so a kernel that mixed two stripes' scales would miss.
        func weights() -> Quantized {
            var w = rng.floats(n * k, scale: 0.1)
            for i in w.indices { w[i] *= Float(1 << (((i % k) / Int8Stripes.run + i / k) % 5)) / 16 }
            return Self.quantize(w.map(Self.bf16Bits), n: n, k: k)
        }
        let z = weights(), z2 = weights()
        let wf = z.values, w2f = z2.values
        let y0 = rng.floats(m * n)
        let xb = try f32Buffer(x), yb = try f32Buffer(y0)
        let lin = try int8Linear(z, n: n, k: k), up = try int8Linear(z2, n: n, k: k)
        try run { e in e.linear(ActView(xb, ld: k), rows: m, lin, ActView(yb, ld: n), flags: flags, gate: flags.isGated ? up : nil) }
        func dot(_ r: Int, _ w: [Float], _ c: Int) -> Double {
            var s = 0.0
            for j in 0..<k { s += Double(x[r * k + j]) * Double(w[c * k + j]) }
            return s
        }
        var want = [Float](repeating: 0, count: m * n)
        for r in 0..<m {
            for c in 0..<n {
                var v = dot(r, wf, c)
                if flags.isGated { v = v / (1 + exp(-v)) * dot(r, w2f, c) }
                if flags.contains(.accumulate) { v += Double(y0[r * n + c]) }
                want[r * n + c] = Float(v)
            }
        }
        let got = yb.floats(count: m * n)
        assertClose(got, want, atol: 2e-4 * Float(k).squareRoot() / 30, "int8 linear m\(m) n\(n) k\(k) flags \(flags.rawValue) layout \(Encoder.int8Layout)")
        return got
    }

    /// The depth decoder's MLP shapes at every row count the frame loop dispatches: one row, the
    /// two rows of a guided step or of an unguided first pair, the four of a guided first pair.
    func testDepthShapes() throws {
        for m in [1, 2, 4] {
            try int8Case(m: m, n: 8192, k: 1024, flags: .siluGate)
            try int8Case(m: m, n: 1024, k: 8192, flags: .accumulate)
        }
        try int8Case(m: 3, n: 1024, k: 1024)
    }

    /// Every layout computes each column from the same lane sums: the bits of the shipped
    /// layout, output counts that are no multiple of the columns per simdgroup included.
    func testLayoutsAreBitIdentical() throws {
        defer { Encoder.int8Layout = .int8Shipped }
        let cases: [(m: Int, n: Int, k: Int, flags: GemvFlags)] = [(2, 1027, 1024, .siluGate), (1, 61, 512, []), (4, 1021, 2048, .accumulate)]
        for c in cases {
            Encoder.int8Layout = .int8Shipped
            let want = try int8Case(m: c.m, n: c.n, k: c.k, flags: c.flags)
            for sg in [1, 2, 4, 8] {
                for columns in Kernels.gemvInt8Columns {
                    Encoder.int8Layout = GemvLayout(simdgroups: sg, columns: columns)
                    let got = try int8Case(m: c.m, n: c.n, k: c.k, flags: c.flags)
                    XCTAssertEqual(got.map(\.bitPattern), want.map(\.bitPattern), "m\(c.m) n\(c.n) k\(c.k) sg\(sg) columns \(columns)")
                }
            }
        }
    }

    /// Bytes the tally charges an int8 projection: the codes and the scales, not the bf16 size.
    func testWeightBytes() throws {
        let z = Self.quantize([UInt16](repeating: Self.bf16Bits(0.5), count: 4 * 512), n: 4, k: 512)
        let lin = try int8Linear(z, n: 4, k: 512)
        XCTAssertEqual(lin.weightBytes, 4 * (512 + 64))
        XCTAssertEqual(Linear(w: lin.w, bias: nil, n: 4, k: 512).weightBytes, 4 * 512 * 2)
    }

    // MARK: whole frame

    /// Teacher-forced on the reference's codes, the int8 kernel and the dequantized weights on
    /// the bf16 kernels give the same guided depth logits to within the bf16 rounding of the
    /// reference weights, and both stay as close to the fp32 goldens as the scheme sweep
    /// measured (cosine 0.9997 at worst over this case's rows; the reference's own bf16 run is
    /// at 0.9996).
    func testFrameAgainstDequantizedWeights() throws {
        guard let goldens = ProcessInfo.processInfo.environment["BREEZE_GOLDENS"] else { throw XCTSkip("set BREEZE_GOLDENS") }
        guard let weights = TestEnvironment.weightsDirectory else { throw XCTSkip("no Breeze-TTS-2 checkpoint") }
        let dir = URL(fileURLWithPath: goldens).appendingPathComponent("design_en")
        var draws: [[Float]] = []
        for format in [DepthWeightFormat.int8, .int8Reference] {
            let parity = try BreezeParity(weights: weights, depth: format)
            let report = try parity.run(caseDirectory: dir)
            let line = try XCTUnwrap(report.tensors.first { $0.name == "tf_sampled_depth_logits" })
            XCTAssertGreaterThan(line.cosine, 0.9995, "\(format): \(line.description)")
            XCTAssertLessThan(line.cosine, 0.999999, "\(format) ran quantized weights: \(line.description)")
            for other in report.tensors where !other.name.contains("depth_logits") { XCTAssertTrue(other.passed, other.description) }
            draws.append(try XCTUnwrap(parity.teacherDraws["sampled_"]).depth)
        }
        let scale = draws[1].map(abs).max()!
        let worst = zip(draws[0], draws[1]).map { abs($0 - $1) }.max()!
        XCTAssertLessThan(worst, 0.02 * scale, "int8 kernel against the dequantized weights")
    }
}
