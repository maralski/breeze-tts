// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import BreezeTTS

/// Shared context and helpers: each kernel runs against a scalar CPU reference on seeded random
/// data. bf16 weights are rounded on the host the same way the GPU reads them, so a comparison
/// isolates kernel arithmetic. Buffers hold NaN wherever a kernel must not read.
class GPUTestCase: XCTestCase {
    static let ctx = try! MetalContext()
    static let kernels = try! Kernels(ctx: ctx)
    var ctx: MetalContext { Self.ctx }
    var kernels: Kernels { Self.kernels }

    struct RNG {
        var state: UInt64
        mutating func bits() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state >> 33
        }
        /// Uniform in [-1, 1).
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24) * 2 - 1
        }
        /// Uniform in [0, 1).
        mutating func unit() -> Float { (next() + 1) / 2 }
        mutating func floats(_ n: Int, scale: Float = 1) -> [Float] { (0..<n).map { _ in next() * scale } }
    }

    static func bf16Bits(_ x: Float) -> UInt16 {
        let u = x.bitPattern
        return UInt16((u &+ 0x7FFF &+ ((u >> 16) & 1)) >> 16)
    }

    static func bf16Round(_ x: Float) -> Float { Float(bitPattern: UInt32(bf16Bits(x)) << 16) }

    func bf16Tensor(_ values: [Float], label: String = "w") throws -> (GPUTensor, [Float]) {
        let b = try ctx.makeBuffer(bf16: values.map(Self.bf16Bits), label: label)
        return (GPUTensor(buffer: b, offset: 0, count: values.count), values.map(Self.bf16Round))
    }

    func f32Buffer(_ values: [Float], label: String = "x") throws -> MTLBuffer {
        try ctx.makeBuffer(floats: values, label: label)
    }

    func nanBuffer(floats count: Int, label: String = "nan") throws -> MTLBuffer {
        try ctx.makeBuffer(floats: [Float](repeating: .nan, count: count), label: label)
    }

    func u32Buffer(_ values: [UInt32], label: String = "u32") throws -> MTLBuffer {
        let b = try ctx.makeBuffer(bytes: values.count * 4, label: label)
        values.withUnsafeBytes { _ = memcpy(b.contents(), $0.baseAddress!, $0.count) }
        return b
    }

    func run(concurrent: Bool = false, _ body: (Encoder) -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        let ce = cb.makeComputeCommandEncoder(dispatchType: concurrent ? .concurrent : .serial)!
        body(Encoder(k: kernels, ce: ce, fenced: concurrent))
        ce.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let e = cb.error { throw BreezeTTSError.gpuFailure("\(e)") }
    }

    /// Fails with the worst element against `atol + rtol * |want|`. NaN on either side fails.
    func assertClose(_ got: [Float], _ want: [Float], atol: Float, rtol: Float = 1e-4, _ what: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(got.count, want.count, what, file: file, line: line)
        var worst = (index: -1, excess: -Float.infinity)
        for i in 0..<min(got.count, want.count) {
            let excess = abs(got[i] - want[i]) - (atol + rtol * abs(want[i]))
            if excess.isNaN { worst = (i, .infinity); break }
            if excess > worst.excess { worst = (i, excess) }
        }
        guard worst.index >= 0 else { return }
        XCTAssertLessThanOrEqual(worst.excess, 0, "\(what): index \(worst.index) got \(got[worst.index]) want \(want[worst.index])", file: file, line: line)
    }

    static func rotateHalf(_ v: [Float], pos: Int, invFreq: [Float]) -> [Float] {
        let half = v.count / 2
        var out = v
        for i in 0..<half {
            let a = Float(pos) * invFreq[i]
            out[i] = v[i] * cos(a) - v[i + half] * sin(a)
            out[i + half] = v[i + half] * cos(a) + v[i] * sin(a)
        }
        return out
    }

    /// softmax(scores) weighted sum of `values` rows.
    static func attend(_ scores: [Float], _ values: [[Float]]) -> [Float] {
        let top = scores.max()!
        let e = scores.map { exp($0 - top) }
        let sum = e.reduce(0, +)
        var out = [Float](repeating: 0, count: values[0].count)
        for (w, v) in zip(e, values) { for d in out.indices { out[d] += w / sum * v[d] } }
        return out
    }

    static func dot(_ a: ArraySlice<Float>, _ b: ArraySlice<Float>) -> Float {
        var s: Float = 0
        for (x, y) in zip(a, b) { s += x * y }
        return s
    }
}
