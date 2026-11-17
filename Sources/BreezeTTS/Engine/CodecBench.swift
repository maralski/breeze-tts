// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Times the streaming codec decoder on the real weights, one frame per command buffer.
///
/// A run resets the decoder and commits `frames` frames back to back, as generation does; a
/// frame's GPU time is the spacing of completion stamps. The same frames then run one at a
/// time with each part of the decoder in its own command buffer, for attribution. Variants
/// (kernel schedules named in BREEZE_CODEC_VARIANTS) take turns within every round so they
/// share machine noise. With `tokenLoop` the frame loop's dispatches precede the codec's in
/// each command buffer and the codec reads that frame's codes: the cost that matters when
/// both share the GPU.
public enum CodecBench {
    public struct Options {
        public var rounds = 7
        public var frames = 40
        public var concurrent = false
        /// Also time the token loop alone and with the codec appended to its command buffers.
        public var tokenLoop = false
        public var guidance: Float = 4
        public init() {}
    }

    /// A kernel schedule: "default", "tMxN" (one tile for every product) or "sgN" (simdgroups
    /// per threadgroup), joined with "+".
    struct Variant {
        let name: String
        var tile: CodecKernels.Tile?
        var simdgroups = 4

        init?(_ name: String) {
            self.name = name
            for part in name.split(separator: "+") where part != "default" {
                if part.hasPrefix("sg"), let n = Int(part.dropFirst(2)), n >= 1, n <= 8 {
                    simdgroups = n
                } else if part.hasPrefix("t"), case let dims = part.dropFirst().split(separator: "x").compactMap({ Int($0) }), dims.count == 2,
                          CodecKernels.tiles.contains(CodecKernels.Tile(tm: dims[0], tn: dims[1])) {
                    tile = CodecKernels.Tile(tm: dims[0], tn: dims[1])
                } else {
                    return nil
                }
            }
        }

        func apply() {
            CodecKernels.tileOverride = tile
            CodecKernels.simdgroups = simdgroups
        }
    }

    struct Samples {
        var gpu: [Double] = [], wall: [Double] = [], syncWall: [Double] = []
        var parts: [String: [Double]] = [:]
        var checksums: Set<UInt64> = []
        var finite = true
        var rms = 0.0
    }

    static func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
    static func span(_ v: [Double]) -> String { String(format: "%.2f [%.2f-%.2f]", median(v), v.min()!, v.max()!) }

    public static func run(directory: URL, options o: Options = Options()) throws {
        let ctx = try MetalContext()
        let kernels = try Kernels(ctx: ctx)
        try ctx.warmUpPipelines()
        let names = (ProcessInfo.processInfo.environment["BREEZE_CODEC_VARIANTS"] ?? "default").split(separator: ",").map(String.init)
        let variants = try names.map { name -> Variant in
            guard let v = Variant(name) else { throw BreezeTTSError.badInput("unknown codec variant \(name)") }
            return v
        }
        let t0 = Date()
        let codecKernels = try CodecKernels(ctx: ctx)
        let weights = try CodecDecoderWeights(try CodecLoader(directory: directory, ctx: ctx))
        print(String(format: "loaded the codec decoder from %@ in %.1f s", directory.lastPathComponent, Date().timeIntervalSince(t0)))
        let decoder = try CodecDecoder(ctx: ctx, kernels: codecKernels, weights: weights, maxFrames: max(2048, o.frames), pcmSlots: o.frames)

        // Seeded codes: the decoder's work does not depend on their values.
        let codes = try ctx.makeBuffer(bytes: o.frames * CodecShape.codebooks * 4, label: "bench.codes")
        var state: UInt64 = 0xC0DEC
        let p = codes.contents().bindMemory(to: UInt32.self, capacity: o.frames * CodecShape.codebooks)
        for i in 0..<(o.frames * CodecShape.codebooks) {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            p[i] = UInt32((state >> 33) % UInt64(CodecShape.codebookSize))
        }

        func encoder(_ cb: MTLCommandBuffer, _ tally: DispatchTally? = nil) -> Encoder {
            Encoder(k: kernels, ce: cb.makeComputeCommandEncoder(dispatchType: o.concurrent ? .concurrent : .serial)!, fenced: o.concurrent, tally: tally)
        }
        func finish(_ cb: MTLCommandBuffer) throws {
            cb.waitUntilCompleted()
            if let e = cb.error { throw BreezeTTSError.gpuFailure("\(e)") }
        }
        func check(_ s: inout Samples) {
            let pcm = decoder.pcm.floats(count: o.frames * CodecShape.samplesPerFrame)
            var hash: UInt64 = 1469598103934665603
            var sum = 0.0
            for v in pcm {
                hash = (hash ^ UInt64(v.bitPattern)) &* 1099511628211
                sum += Double(v) * Double(v)
                if !v.isFinite { s.finite = false }
            }
            s.checksums.insert(hash)
            s.rms = (sum / Double(pcm.count)).squareRoot()
        }

        // Dispatch and work counts of a steady-state frame (frame 1; frame 0 also zeroes the histories).
        decoder.reset()
        _ = decoder.append(codes: codes, codesOffset: 0)
        CodecKernels.multiplyAdds = 0
        let tally = DispatchTally()
        var partWork: [(name: String, dispatches: Int, multiplyAdds: Int)] = []
        do {
            let step = decoder.append(codes: codes, codesOffset: 64)
            let cb = ctx.queue.makeCommandBuffer()!
            let e = encoder(cb, tally)
            for part in decoder.lastParts {
                let before = CodecKernels.multiplyAdds
                for d in step.dispatches[part.dispatches] { d(e) }
                partWork.append((part.name, part.dispatches.count, CodecKernels.multiplyAdds - before))
            }
            e.ce.endEncoding()
        }
        let frameMultiplyAdds = CodecKernels.multiplyAdds

        var samples = [Samples](repeating: Samples(), count: variants.count)
        for round in 0..<(o.rounds + 1) {
            let order = round % 2 == 0 ? Array(variants.indices) : variants.indices.reversed()
            for i in order {
                variants[i].apply()
                decoder.reset()
                let w0 = Date()
                var buffers: [MTLCommandBuffer] = []
                for f in 0..<o.frames {
                    let step = decoder.append(codes: codes, codesOffset: f * 64)
                    let cb = ctx.queue.makeCommandBuffer()!
                    let e = encoder(cb)
                    for d in step.dispatches { d(e) }
                    e.ce.endEncoding()
                    cb.commit()
                    buffers.append(cb)
                }
                for cb in buffers { try finish(cb) }
                let wall = Date().timeIntervalSince(w0) / Double(o.frames) * 1e3
                let gpu = (buffers.last!.gpuEndTime - buffers[0].gpuEndTime) / Double(o.frames - 1) * 1e3
                check(&samples[i])

                decoder.reset()
                var parts: [String: Double] = [:]
                let s0 = Date()
                for f in 0..<o.frames {
                    let step = decoder.append(codes: codes, codesOffset: f * 64)
                    var pending: [(String, MTLCommandBuffer)] = []
                    for part in decoder.lastParts {
                        let cb = ctx.queue.makeCommandBuffer()!
                        let e = encoder(cb)
                        for d in step.dispatches[part.dispatches] { d(e) }
                        e.ce.endEncoding()
                        cb.commit()
                        pending.append((part.name, cb))
                    }
                    for (name, cb) in pending {
                        try finish(cb)
                        if f > 0 { parts[name, default: 0] += cb.gpuEndTime - cb.gpuStartTime }
                    }
                }
                let syncWall = Date().timeIntervalSince(s0) / Double(o.frames) * 1e3
                check(&samples[i])
                // Round 0 warms the GPU up and is not kept.
                if round > 0 {
                    samples[i].gpu.append(gpu)
                    samples[i].wall.append(wall)
                    samples[i].syncWall.append(syncWall)
                    for (name, t) in parts { samples[i].parts[name, default: []].append(t / Double(o.frames - 1) * 1e3) }
                }
            }
        }
        Variant("default")!.apply()

        var floors: [Double] = []
        let scratch = try ctx.makeBuffer(bytes: 64, label: "bench.scratch")
        for _ in 0..<5 {
            let cb = ctx.queue.makeCommandBuffer()!
            let e = encoder(cb)
            for _ in 0..<tally.dispatches { e.copy(ActView(scratch, ld: 1), count: 1, to: ActView(scratch, offset: 32, ld: 1)) }
            e.ce.endEncoding()
            cb.commit()
            try finish(cb)
            floors.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
        }

        print("codec decoder, streaming one frame (1920 samples) per command buffer; \(o.frames) frames per run, median [min-max] of \(o.rounds) interleaved rounds after one warm-up round, \(o.concurrent ? "concurrent encoder with barriers" : "serial encoder")")
        print(String(format: "a frame is %d dispatches, %.2f G multiply-adds, %.0f MB of weights read; the same number of empty dispatches takes %.2f ms",
                     tally.dispatches, Double(frameMultiplyAdds) / 1e9, Double(tally.weightBytes) / 1e6, median(floors)))
        let allSums = samples.reduce(into: Set<UInt64>()) { $0.formUnion($1.checksums) }
        for (v, s) in zip(variants, samples) {
            let gpu = median(s.gpu)
            print(String(format: "%@: gpu %@ ms per frame, wall %@ ms, one-at-a-time wall %@ ms; %.0f G multiply-adds/s; %.2f of the 15 ms budget, %.3f of an 80 ms frame",
                         v.name, span(s.gpu), span(s.wall), span(s.syncWall), Double(frameMultiplyAdds) / gpu / 1e6, gpu / 15, gpu / 80))
            print("  work check: samples finite \(s.finite), rms \(String(format: "%.4f", s.rms)), pcm identical in every run of this variant \(s.checksums.count == 1), across variants \(allSums.count == 1)")
            let total = partWork.reduce(0.0) { $0 + median(s.parts[$1.name] ?? [0]) }
            for part in partWork {
                let t = median(s.parts[part.name] ?? [0])
                print(String(format: "  %@ %6.2f ms %5.1f%%  %3d dispatches  %7.1f M multiply-adds  %5.0f G/s", part.name.padding(toLength: 24, withPad: " ", startingAt: 0),
                             t, t / total * 100, part.dispatches, Double(part.multiplyAdds) / 1e6, t > 0 ? Double(part.multiplyAdds) / t / 1e6 : 0))
            }
            print(String(format: "  parts sum to %.2f ms when each runs alone in its own command buffer", total))
        }

        if o.tokenLoop { try tokenLoop(directory: directory, ctx: ctx, kernels: kernels, decoder: decoder, options: o) }
    }

    /// The frame loop alone against the frame loop with the codec decoding each frame's codes in
    /// the same command buffer, interleaved. The difference is the codec's cost when it shares
    /// the GPU.
    static func tokenLoop(directory: URL, ctx: MetalContext, kernels: Kernels, decoder: CodecDecoder, options o: Options) throws {
        let weights = try FrameWeights(try ShardedCheckpoint(directory: directory), ctx: ctx)
        var settings = FrameSettings()
        settings.guidance = o.guidance
        let loop = try FrameLoop(ctx: ctx, weights: weights, sequences: 2, maxFrames: o.frames)
        let kv = loop.backbone.cache
        for layer in kv.layers {
            for t in [layer.k, layer.v] { GemvBench.fillF32(t.buffer, offset: t.offset, count: kv.sequences * layer.seqStride) }
        }
        func start() {
            loop.begin(positions: [300, 260])
            GemvBench.fillF32(loop.backboneInput, offset: 0, count: loop.sequences * BreezeShape.embedWidth)
            var s: UInt64 = 0xB5EE2E
            for f in 0..<o.frames {
                loop.setUniforms(frame: f, (0..<BreezeShape.codebooks).map { _ in
                    s = s &* 6364136223846793005 &+ 1442695040888963407
                    return Float(s >> 40) / Float(1 << 24)
                })
            }
            decoder.reset()
        }
        var gpu: [[Double]] = [[], []], wall: [[Double]] = [[], []]
        var finite = true, sums: Set<UInt64> = []
        for round in 0..<(o.rounds + 1) {
            for withCodec in (round % 2 == 0 ? [false, true] : [true, false]) {
                start()
                let w0 = Date()
                var buffers: [MTLCommandBuffer] = []
                for f in 0..<o.frames {
                    let cb = ctx.queue.makeCommandBuffer()!
                    let e = Encoder(k: kernels, ce: cb.makeComputeCommandEncoder(dispatchType: o.concurrent ? .concurrent : .serial)!, fenced: o.concurrent)
                    loop.encodeFrame(e, frame: f, settings: settings)
                    if withCodec {
                        for d in decoder.append(codes: loop.codes, codesOffset: f * BreezeShape.codebooks * 4).dispatches { d(e) }
                    }
                    e.ce.endEncoding()
                    cb.commit()
                    buffers.append(cb)
                }
                for cb in buffers {
                    cb.waitUntilCompleted()
                    if let e = cb.error { throw BreezeTTSError.gpuFailure("\(e)") }
                }
                if withCodec {
                    var hash: UInt64 = 1469598103934665603
                    for v in decoder.pcm.floats(count: o.frames * CodecShape.samplesPerFrame) {
                        hash = (hash ^ UInt64(v.bitPattern)) &* 1099511628211
                        if !v.isFinite { finite = false }
                    }
                    sums.insert(hash)
                }
                if round > 0 {
                    gpu[withCodec ? 1 : 0].append((buffers.last!.gpuEndTime - buffers[0].gpuEndTime) / Double(o.frames - 1) * 1e3)
                    wall[withCodec ? 1 : 0].append(Date().timeIntervalSince(w0) / Double(o.frames) * 1e3)
                }
            }
        }
        print("token loop (guided, backbone at positions [300, 260]) with the codec in the same command buffer, \(o.frames) frames per run, \(o.rounds) interleaved rounds:")
        print(String(format: "  token loop alone: gpu %@ ms per frame, wall %@ ms", span(gpu[0]), span(wall[0])))
        print(String(format: "  token loop + codec: gpu %@ ms per frame, wall %@ ms; realtime factor %.2f of an 80 ms frame", span(gpu[1]), span(wall[1]), median(gpu[1]) / 80))
        print(String(format: "  codec adds %.2f ms per frame; work check: samples finite \(finite), pcm identical in every run \(sums.count == 1), loop status clean \(loop.statusWord == 0)",
                     median(gpu[1]) - median(gpu[0])))
    }
}
