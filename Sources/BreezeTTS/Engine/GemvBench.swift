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

    /// A `GemvLayout` spec ("sgN", "sgNrC"), "gemm8": the matrix-unit kernel even at one or
    /// two rows, or "i8:sgNrC": the int8 kernel in that layout, on the shapes it accepts, the
    /// pool read as int8 with every scale 1/64.
    struct Variant {
        let label: String
        var layout = GemvLayout.shipped
        var scalarRowsMax = 2
        var int8 = false

        init(_ spec: String) throws {
            label = spec
            if spec == "gemm8" { scalarRowsMax = 0; return }
            int8 = spec.hasPrefix("i8:")
            guard let l = GemvLayout(int8 ? String(spec.dropFirst(3)) : spec, columns: int8 ? Kernels.gemvInt8Columns : Kernels.gemvColumns) else {
                throw BreezeTTSError.badInput("unknown bench variant \(spec)")
            }
            layout = l
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
        let ones = try ctx.makeBuffer(bytes: 8192 * Int8Stripes.lanes * 2, label: "bench.scales")
        let half = ones.contents().bindMemory(to: Float16.self, capacity: 8192 * Int8Stripes.lanes)
        for i in 0..<(8192 * Int8Stripes.lanes) { half[i] = 1 / 64 }
        let scales = GPUTensor(buffer: ones, offset: 0, count: 8192 * Int8Stripes.lanes)
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
        print(String(format: "%d dispatches per shape, median of %d interleaved command buffers, us per dispatch (GB/s of the weight bytes that variant reads); empty dispatch floor %.1f us", repeats, rounds, floors.sorted()[rounds / 2]))
        print("shape              M     N     K    MB  " + variants.map { $0.label.padding(toLength: 18, withPad: " ", startingAt: 0) }.joined())
        defer { Encoder.gemvLayoutOverride = nil; Encoder.scalarRowsMax = 2; Encoder.int8Layout = .int8Shipped }
        for s in shapes {
            let fitsInt8 = s.k % Int8Stripes.period == 0
            func encodeAll(_ v: Variant) -> (Encoder) -> Void {
                return { e in
                    Encoder.gemvLayoutOverride = v.layout
                    Encoder.scalarRowsMax = v.scalarRowsMax
                    Encoder.int8Layout = v.int8 ? v.layout : .int8Shipped
                    // An int8 matrix is half the bytes, so the walk through the pool is half as fast.
                    let one = s.n * s.k * (v.int8 ? 1 : 2), step = one * (s.flags.isGated ? 2 : 1)
                    var offset = 0
                    for _ in 0..<repeats {
                        if offset + step > poolBytes { offset = 0 }
                        let lin = Linear(w: GPUTensor(buffer: pool, offset: offset, count: s.n * s.k), bias: nil, n: s.n, k: s.k, scales: v.int8 ? scales : nil)
                        let gate = s.flags.isGated ? Linear(w: GPUTensor(buffer: pool, offset: offset + one, count: s.n * s.k), bias: nil, n: s.n, k: s.k, scales: v.int8 ? scales : nil) : nil
                        e.linear(ActView(x, ld: s.k), rows: s.m, lin, ActView(y, ld: s.n), flags: s.flags, gate: gate)
                        offset += step
                    }
                }
            }
            var samples = [[Double]](repeating: [], count: variants.count)
            for r in 0..<(rounds + 1) {
                for i in (r % 2 == 0 ? Array(variants.indices) : Array(variants.indices.reversed())) where fitsInt8 || !variants[i].int8 {
                    let t = try time(encodeAll(variants[i]))
                    if r > 0 { samples[i].append(t) }
                }
            }
            let mb = Double(s.bytes) / 1e6
            let cells = zip(variants, samples).map { variant, v -> String in
                guard !v.isEmpty else { return "        -          " }
                let med = v.sorted()[rounds / 2]
                return String(format: "%9.1f (%4.0f)   ", med, mb * (variant.int8 ? 0.5 : 1) * 1e3 / med)
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
