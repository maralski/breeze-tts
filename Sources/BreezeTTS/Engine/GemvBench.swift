// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Isolated timing of the decode-time projection shapes (adapted from MingTTS): `repeats`
/// back-to-back dispatches per shape, each reading a fresh slice of a 1 GiB weight pool so no
/// cache level serves the weights twice, under each variant of BREEZE_BENCH_VARIANTS, interleaved.
public enum GemvBench {
    struct Shape {
        let label: String
        let m, n, k: Int
        let flags: GemvFlags
        var bytes: Int { n * k * 2 * (flags.isGated ? 2 : 1) }
    }

    static let shapes: [Shape] = [1, 2].flatMap { m in [
        Shape(label: "backbone qkv", m: m, n: 4096, k: 2048, flags: []),
        Shape(label: "backbone o", m: m, n: 2048, k: 2048, flags: .accumulate),
        Shape(label: "backbone gate|up", m: m, n: 6144, k: 2048, flags: .siluGate),
        Shape(label: "backbone down", m: m, n: 2048, k: 6144, flags: .accumulate),
        Shape(label: "lm head", m: m, n: 2052, k: 2048, flags: []),
        Shape(label: "depth projector", m: m, n: 1024, k: 2048, flags: []),
        Shape(label: "depth qkv", m: m, n: 1536, k: 1024, flags: []),
        Shape(label: "depth o", m: m, n: 1024, k: 1024, flags: .accumulate),
        Shape(label: "depth gate|up", m: m, n: 8192, k: 1024, flags: .siluGate),
        Shape(label: "depth down", m: m, n: 1024, k: 8192, flags: .accumulate),
        Shape(label: "codebook head", m: m, n: 2051, k: 1024, flags: []),
    ] }

    /// "sgN": N output columns per threadgroup of the scalar kernel. "gemm8": the matrix-unit
    /// kernel even at one or two rows.
    struct Variant {
        let label: String
        var simdgroups = 4
        var scalarRowsMax = 2

        init(_ spec: String) throws {
            label = spec
            if spec == "gemm8" { scalarRowsMax = 0 } else if spec.hasPrefix("sg"), let v = Int(spec.dropFirst(2)), [1, 2, 4, 8, 16].contains(v) { simdgroups = v } else {
                throw BreezeTTSError.badInput("unknown bench variant \(spec)")
            }
        }
    }

    public static func run(repeats: Int = 100, rounds: Int = 5) throws {
        let ctx = try MetalContext()
        let kernels = try Kernels(ctx: ctx)
        let poolBytes = 1 << 30
        let pool = try ctx.makeBuffer(bytes: poolBytes, label: "bench.pool")
        fillBF16(pool)
        let x = try ctx.makeBuffer(bytes: 16 * 8192 * 4, label: "bench.x")
        fillF32(x, offset: 0, count: 16 * 8192)
        let y = try ctx.makeBuffer(bytes: 16 * 8192 * 4, label: "bench.y")
        let specs = (ProcessInfo.processInfo.environment["BREEZE_BENCH_VARIANTS"] ?? "sg1,sg2,sg4,sg8").split(separator: ",").map(String.init)
        let variants = try specs.map(Variant.init)

        func time(_ body: (Encoder) -> Void) throws -> Double {
            let cb = ctx.queue.makeCommandBuffer()!
            let ce = cb.makeComputeCommandEncoder()!
            body(Encoder(k: kernels, ce: ce))
            ce.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { throw BreezeTTSError.gpuFailure("\(e)") }
            return (cb.gpuEndTime - cb.gpuStartTime) / Double(repeats) * 1e6
        }

        var floors: [Double] = []
        for _ in 0..<rounds {
            floors.append(try time { e in for _ in 0..<repeats { e.copy(ActView(x, ld: 1), count: 1, to: ActView(y, ld: 1)) } })
        }
        print(String(format: "%d dispatches per shape, median of %d interleaved command buffers, us per dispatch (GB/s of weights); empty dispatch floor %.1f us", repeats, rounds, floors.sorted()[rounds / 2]))
        print("shape              M     N     K    MB  " + variants.map { $0.label.padding(toLength: 18, withPad: " ", startingAt: 0) }.joined())
        defer { Encoder.gemvSimdgroups = 4; Encoder.scalarRowsMax = 2 }
        for s in shapes {
            func encodeAll(_ v: Variant) -> (Encoder) -> Void {
                return { e in
                    Encoder.gemvSimdgroups = v.simdgroups
                    Encoder.scalarRowsMax = v.scalarRowsMax
                    var offset = 0
                    for _ in 0..<repeats {
                        if offset + s.bytes > poolBytes { offset = 0 }
                        let lin = Linear(w: GPUTensor(buffer: pool, offset: offset, count: s.n * s.k), bias: nil, n: s.n, k: s.k)
                        let gate = s.flags.isGated ? GPUTensor(buffer: pool, offset: offset + s.n * s.k * 2, count: s.n * s.k) : nil
                        e.linear(ActView(x, ld: s.k), rows: s.m, lin, ActView(y, ld: s.n), flags: s.flags, gate: gate)
                        offset += s.bytes
                    }
                }
            }
            var samples = [[Double]](repeating: [], count: variants.count)
            for r in 0..<(rounds + 1) {
                for i in (r % 2 == 0 ? Array(variants.indices) : Array(variants.indices.reversed())) {
                    let t = try time(encodeAll(variants[i]))
                    if r > 0 { samples[i].append(t) }
                }
            }
            let mb = Double(s.bytes) / 1e6
            let cells = samples.map { v -> String in
                let med = v.sorted()[rounds / 2]
                return String(format: "%9.1f (%4.0f)   ", med, mb * 1e3 / med)
            }
            print(s.label.padding(toLength: 17, withPad: " ", startingAt: 0) + String(format: "%3d %5d %5d %5.1f  ", s.m, s.n, s.k, mb) + cells.joined())
        }
    }

    /// bf16 values in [-0.5, -0.125] and [0.125, 0.5): a valid exponent range, both signs.
    static func fillBF16(_ b: MTLBuffer) {
        let n = b.length / 2
        let p = b.contents().bindMemory(to: UInt16.self, capacity: n)
        var s: UInt64 = 0x9E3779B97F4A7C15
        for i in 0..<n {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            p[i] = (UInt16(truncatingIfNeeded: s >> 48) & 0x81FF) | 0x3E00
        }
    }

    /// Floats in [-1, 1), the same sequence every call.
    static func fillF32(_ b: MTLBuffer, offset: Int, count: Int) {
        let p = (b.contents() + offset).bindMemory(to: Float.self, capacity: count)
        var s: UInt64 = 12345
        for i in 0..<count {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            p[i] = Float(s >> 40) / Float(1 << 24) * 2 - 1
        }
    }
}
