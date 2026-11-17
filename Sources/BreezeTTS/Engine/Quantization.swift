// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// How the depth decoder's MLP weights sit in memory.
enum DepthWeightFormat: Equatable {
    /// The checkpoint's own bf16.
    case bf16
    /// `Int8Stripes`, quantized from the bf16 at load.
    case int8
    /// The `int8` weights multiplied back out and rounded to bf16, for the bf16 kernels: the
    /// same model to within bf16 rounding of each weight (2^-9 relative, against a quantization
    /// step of about 2^-7), with none of the int8 kernel. The quality study's control for the
    /// kernel.
    case int8Reference
}

/// Symmetric int8 weights for the one-row projection kernel: w[n, k] = q[n, k] * scale[n, lane(k)].
///
/// The kernel's 32 lanes each load 16 int8 at a time, so lane l reads k = 512 i + 16 l ..< + 16
/// for i = 0, 1, ... That set of K / 32 weights is the lane's stripe, and each stripe of each row
/// has one fp16 scale, the stripe's largest magnitude over 127. A lane then scales its whole sum
/// once. tools/quant/depth_quant.py holds the same quantizer in Python ("stripeall/free/max"),
/// and its `checksum` command and `QuantizationTests` pin the two to the same bytes.
enum Int8Stripes {
    static let lanes = 32
    static let run = 16
    static let period = lanes * run

    /// Largest magnitude a scale may map to; -128 is never produced.
    static let limit: Float = 127

    @inline(__always) static func value(_ bits: UInt16) -> Float { Float(bitPattern: UInt32(bits) << 16) }

    /// Quantizes one row of `k` bf16 weights into `q` (k values) and `scales` (`lanes` values).
    static func quantize(row: UnsafePointer<UInt16>, k: Int, q: UnsafeMutablePointer<Int8>, scales: UnsafeMutablePointer<Float16>) {
        precondition(k % period == 0, "int8 stripes need K % \(period) == 0, got \(k)")
        for lane in 0..<lanes {
            var amax: Float = 0
            var i = lane * run
            while i < k {
                for j in i..<(i + run) { amax = max(amax, abs(value(row[j]))) }
                i += period
            }
            let scale = Float16(amax / limit)
            scales[lane] = scale
            // A stripe too small for an fp16 scale (largest weight under 8e-6) becomes zeros.
            let inverse = scale > 0 ? 1 / Float(scale) : 0
            let s = Float(scale)
            i = lane * run
            while i < k {
                for j in i..<(i + run) {
                    let x = inverse > 0 ? value(row[j]) / s : 0
                    // Half away from zero, written as the Python quantizer writes it.
                    let r = min((abs(x) + 0.5).rounded(.down), limit)
                    q[j] = Int8(x < 0 ? -r : r)
                }
                i += period
            }
        }
    }

    /// q * scale as fp32, exact.
    @inline(__always) static func weight(_ q: Int8, _ scale: Float16) -> Float { Float(q) * Float(scale) }

    /// Round to nearest even, as the checkpoint's own bf16 was rounded.
    @inline(__always) static func bf16Bits(_ x: Float) -> UInt16 {
        let u = x.bitPattern
        return UInt16(truncatingIfNeeded: (u &+ 0x7FFF &+ ((u >> 16) & 1)) >> 16)
    }

    /// Rows `0 ..< n` of a bf16 [n, k] weight, quantized on every core.
    static func quantize(_ src: UnsafePointer<UInt16>, n: Int, k: Int, q: UnsafeMutablePointer<Int8>, scales: UnsafeMutablePointer<Float16>) {
        nonisolated(unsafe) let (src, q, scales) = (src, q, scales)
        let chunk = max(1, 65536 / k)
        DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { c in
            for r in (c * chunk)..<min(n, (c + 1) * chunk) {
                quantize(row: src + r * k, k: k, q: q + r * k, scales: scales + r * lanes)
            }
        }
    }

    /// The bf16 [n, k] weight the quantized form of `src` stands for, rounded to bf16.
    static func reference(_ src: UnsafePointer<UInt16>, n: Int, k: Int, into dst: UnsafeMutablePointer<UInt16>) {
        nonisolated(unsafe) let (src, dst) = (src, dst)
        let chunk = max(1, 65536 / k)
        DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { c in
            var q = [Int8](repeating: 0, count: k)
            var scales = [Float16](repeating: 0, count: lanes)
            for r in (c * chunk)..<min(n, (c + 1) * chunk) {
                quantize(row: src + r * k, k: k, q: &q, scales: &scales)
                let out = dst + r * k
                for j in 0..<k { out[j] = bf16Bits(weight(q[j], scales[(j / run) % lanes])) }
            }
        }
    }
}
