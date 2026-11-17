// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Times the real per-frame schedule over the real checkpoint weights, before any parity work:
/// `FrameLoop.encodeFrame` with and without guidance, backbone caches pre-filled to a few hundred
/// positions with synthetic keys and values. The two modes take turns within each round so they
/// share machine noise. A run of frames is committed back to back, as generation does, and a
/// frame's GPU time is the spacing of completion stamps; the same frames are then run one at a
/// time with a wait, and split into their two stages for attribution.
///
/// BREEZE_FRAME_VARIANTS names kernel schedules to compare in the same process, separated by
/// ";": "default", or rules "MxNxK=sgSrC" joined by ",", each giving one scalar-projection shape
/// a `GemvLayout`; "*=sgSrC" gives it to every decode shape, and later rules override it. Every variant runs plain and guided in every round, in alternating order;
/// BREEZE_FRAME_ONLY=guided (or plain) times that mode alone, for more rounds in the same time.
public enum FrameBench {
    public struct Options {
        public var rounds = 7
        public var frames = 12
        /// Concurrent encoder with a barrier before every dispatch, instead of a serial encoder.
        public var concurrent = false
        public var guidance: Float = 4
        public var positions = [300, 260]
        public init() {}
    }

    /// BREEZE_WEIGHTS, else the first Breeze-TTS-2 snapshot in the Hugging Face cache.
    public static func defaultWeights(environment env: [String: String] = ProcessInfo.processInfo.environment) -> URL? {
        if let path = env["BREEZE_WEIGHTS"] { return URL(fileURLWithPath: path) }
        let hub = env["HF_HUB_CACHE"].map(URL.init(fileURLWithPath:))
            ?? URL(fileURLWithPath: env["HF_HOME"] ?? NSHomeDirectory() + "/.cache/huggingface").appendingPathComponent("hub")
        let snapshots = hub.appendingPathComponent("models--BreezeBlue--Breeze-TTS-2/snapshots")
        let found = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: nil)) ?? []
        return found.first { FileManager.default.fileExists(atPath: $0.appendingPathComponent(ShardedCheckpoint.indexName).path) }
    }

    /// "default", or "MxNxK=sgSrC,..." rules.
    static func overrides(_ spec: String) throws -> [GemvShape: GemvLayout] {
        var out: [GemvShape: GemvLayout] = [:]
        for rule in spec.split(separator: ",") where rule != "default" {
            let sides = rule.split(separator: "="), dims = sides[0].split(separator: "x").compactMap { Int($0) }
            if sides.count == 2, sides[0] == "*", let layout = GemvLayout(String(sides[1])) {
                for s in GemvBench.shapes { out[GemvShape(m: s.m, n: s.n, k: s.k)] = layout }
                continue
            }
            guard sides.count == 2, dims.count == 3, let layout = GemvLayout(String(sides[1])) else { throw BreezeTTSError.badInput("unknown schedule rule \(rule)") }
            out[GemvShape(m: dims[0], n: dims[1], k: dims[2])] = layout
        }
        return out
    }

    struct Mode {
        let name: String
        var overrides: [GemvShape: GemvLayout] = [:]
        let settings: FrameSettings
        let loop: FrameLoop
        let positions: [Int]
        var gpu: [Double] = [], wall: [Double] = [], syncWall: [Double] = [], backbone: [Double] = [], codes: [Double] = []
        var firstCodes: [[Int]]?
        var codesStable = true, codesInRange = true, statusClean = true
        var endOfSpeech = 0
    }

    static func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
    static func span(_ v: [Double]) -> String { String(format: "%.1f [%.1f-%.1f]", median(v), v.min()!, v.max()!) }

    public static func run(directory: URL, options o: Options = Options()) throws {
        let ctx = try MetalContext()
        let kernels = try Kernels(ctx: ctx)
        try ctx.warmUpPipelines()
        if let split = ProcessInfo.processInfo.environment["BREEZE_CAUSAL_SPLIT"].flatMap(Int.init), Kernels.causalSplits.contains(split) {
            Encoder.causalDecodeSplit = split
        }
        let t0 = Date()
        let weights = try FrameWeights(try ShardedCheckpoint(directory: directory), ctx: ctx)
        print(String(format: "loaded backbone, depth decoder, heads and embedding from %@ in %.1f s", directory.lastPathComponent, Date().timeIntervalSince(t0)))

        var guided = FrameSettings()
        guided.guidance = o.guidance
        let plainLoop = try FrameLoop(ctx: ctx, weights: weights, sequences: 1, maxFrames: o.frames)
        let guidedLoop = try FrameLoop(ctx: ctx, weights: weights, sequences: 2, maxFrames: o.frames)
        let specs = (ProcessInfo.processInfo.environment["BREEZE_FRAME_VARIANTS"] ?? "default").split(separator: ";").map(String.init)
        var modes: [Mode] = []
        for spec in specs {
            let rules = try overrides(spec)
            let tag = specs.count == 1 ? "" : " [\(spec)]"
            modes.append(Mode(name: "plain" + tag, overrides: rules, settings: FrameSettings(), loop: plainLoop, positions: [o.positions[0]]))
            modes.append(Mode(name: "guided" + tag, overrides: rules, settings: guided, loop: guidedLoop, positions: o.positions))
        }
        defer { Encoder.gemvShapeOverrides = [:] }
        for m in modes.prefix(2) {
            // Synthetic prompt: every cached key and value the frames will read.
            let kv = m.loop.backbone.cache
            for layer in kv.layers {
                for t in [layer.k, layer.v] { GemvBench.fillF32(t.buffer, offset: t.offset, count: kv.sequences * layer.seqStride) }
            }
        }

        func encoder(_ cb: MTLCommandBuffer, _ tally: DispatchTally? = nil) -> Encoder {
            Encoder(k: kernels, ce: cb.makeComputeCommandEncoder(dispatchType: o.concurrent ? .concurrent : .serial)!, fenced: o.concurrent, tally: tally)
        }
        func finish(_ cb: MTLCommandBuffer) throws {
            cb.waitUntilCompleted()
            if let e = cb.error { throw BreezeTTSError.gpuFailure("\(e)") }
        }
        /// The same starting state every time: prompt length, first input, uniforms, empty history.
        func start(_ m: Mode) {
            m.loop.begin(positions: m.positions)
            GemvBench.fillF32(m.loop.backboneInput, offset: 0, count: m.loop.sequences * BreezeShape.embedWidth)
            var s: UInt64 = 0xB5EE2E
            for f in 0..<o.frames {
                m.loop.setUniforms(frame: f, (0..<BreezeShape.codebooks).map { _ in
                    s = s &* 6364136223846793005 &+ 1442695040888963407
                    return Float(s >> 40) / Float(1 << 24)
                })
            }
        }
        func check(_ m: inout Mode) {
            let codes = (0..<o.frames).map { m.loop.frameCodes($0) }
            if let first = m.firstCodes { m.codesStable = m.codesStable && first == codes } else { m.firstCodes = codes }
            m.statusClean = m.statusClean && m.loop.statusWord == 0
            m.codesInRange = m.codesInRange && codes.allSatisfy { f in f.dropFirst().allSatisfy { $0 < 2048 } && (f[0] < 2048 || f[0] == BreezeShape.endOfSpeech) }
            m.endOfSpeech = codes.filter { $0[0] == BreezeShape.endOfSpeech }.count
        }

        let only = ProcessInfo.processInfo.environment["BREEZE_FRAME_ONLY"]
        var counts: [(String, DispatchTally)] = []
        for m in modes.prefix(2) {
            start(m)
            let tally = DispatchTally()
            let cb = ctx.queue.makeCommandBuffer()!
            let e = encoder(cb, tally)
            m.loop.encodeFrame(e, frame: 0, settings: m.settings)
            e.ce.endEncoding()
            cb.commit()
            try finish(cb)
            counts.append((m.name, tally))
        }

        for round in 0..<(o.rounds + 1) {
            for i in (round % 2 == 0 ? Array(modes.indices) : modes.indices.reversed()) where only == nil || modes[i].name.hasPrefix(only!) {
                Encoder.gemvShapeOverrides = modes[i].overrides
                // Frames committed back to back.
                start(modes[i])
                let w0 = Date()
                var buffers: [MTLCommandBuffer] = []
                for f in 0..<o.frames {
                    let cb = ctx.queue.makeCommandBuffer()!
                    let e = encoder(cb)
                    modes[i].loop.encodeFrame(e, frame: f, settings: modes[i].settings)
                    e.ce.endEncoding()
                    cb.commit()
                    buffers.append(cb)
                }
                for cb in buffers { try finish(cb) }
                let wall = Date().timeIntervalSince(w0) / Double(o.frames) * 1e3
                let gpu = (buffers.last!.gpuEndTime - buffers[0].gpuEndTime) / Double(o.frames - 1) * 1e3
                check(&modes[i])

                // One frame at a time, waiting for each, the two stages in separate command buffers.
                start(modes[i])
                let s0 = Date()
                var backbone = 0.0, codes = 0.0
                for f in 0..<o.frames {
                    let a = ctx.queue.makeCommandBuffer()!
                    let ea = encoder(a)
                    modes[i].loop.encodeBackboneStep(ea)
                    ea.ce.endEncoding()
                    a.commit()
                    let b = ctx.queue.makeCommandBuffer()!
                    let eb = encoder(b)
                    modes[i].loop.encodeCodes(eb, frame: f, settings: modes[i].settings)
                    eb.ce.endEncoding()
                    b.commit()
                    try finish(a)
                    try finish(b)
                    backbone += a.gpuEndTime - a.gpuStartTime
                    codes += b.gpuEndTime - b.gpuStartTime
                }
                let syncWall = Date().timeIntervalSince(s0) / Double(o.frames) * 1e3
                check(&modes[i])
                // Round 0 warms the GPU up and is not kept.
                if round > 0 {
                    modes[i].gpu.append(gpu)
                    modes[i].wall.append(wall)
                    modes[i].syncWall.append(syncWall)
                    modes[i].backbone.append(backbone / Double(o.frames) * 1e3)
                    modes[i].codes.append(codes / Double(o.frames) * 1e3)
                }
            }
        }

        Encoder.gemvShapeOverrides = [:]
        // What the same number of dispatches costs when each does nothing.
        let dispatches = counts[0].1.dispatches
        var floors: [Double] = []
        let scratch = try ctx.makeBuffer(bytes: 64, label: "bench.scratch")
        for _ in 0..<5 {
            let cb = ctx.queue.makeCommandBuffer()!
            let e = encoder(cb)
            for _ in 0..<dispatches { e.copy(ActView(scratch, ld: 1), count: 1, to: ActView(scratch, offset: 32, ld: 1)) }
            e.ce.endEncoding()
            cb.commit()
            try finish(cb)
            floors.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
        }

        print("\(o.frames) frames per run, median [min-max] of \(o.rounds) interleaved rounds after one warm-up round, \(o.concurrent ? "concurrent encoder with barriers" : "serial encoder"), backbone at positions \(o.positions)")
        for (i, m) in modes.enumerated() where !m.gpu.isEmpty {
            let tally = counts[i % 2].1
            let gb = Double(tally.weightBytes) / 1e9
            print(String(format: "frame %@: gpu %@ ms, wall %@ ms, one-at-a-time wall %@ ms, stages backbone %@ + codes %@ ms",
                         m.name, span(m.gpu), span(m.wall), span(m.syncWall), span(m.backbone), span(m.codes)))
            print(String(format: "  %d dispatches, %.2f GB of weights read, %.0f GB/s over the frame; realtime factor %.2f of an 80 ms frame",
                         tally.dispatches, gb, gb / (median(m.gpu) / 1e3), median(m.gpu) / 80))
            let sameAsFirst = m.firstCodes.map { $0 == modes[i % 2].firstCodes } ?? false
            if i >= 2 {
                // Each round runs every variant once, so a round's difference shares its noise.
                let first = modes[i % 2]
                let d = zip(m.gpu, first.gpu).map { $0 - $1 }, c = zip(m.codes, first.codes).map { $0 - $1 }
                print(String(format: "  against the first variant, per round: gpu %+.2f [%+.2f to %+.2f] ms, faster in %d of %d; codes stage %+.2f [%+.2f to %+.2f] ms, faster in %d of %d",
                             median(d), d.min()!, d.max()!, d.filter { $0 < 0 }.count, d.count, median(c), c.min()!, c.max()!, c.filter { $0 < 0 }.count, c.count))
            }
            print("  work check: status clean \(m.statusClean), codes in range \(m.codesInRange), codes identical in every run \(m.codesStable) and to the first variant's \(sameAsFirst), end-of-speech frames \(m.endOfSpeech) of \(o.frames)")
        }
        // Each kernel that reads no projection weights, alone: as many dispatches as a frame
        // issues, in one command buffer.
        func part(_ name: String, _ loop: FrameLoop, count: Int, _ body: (Encoder, Int) -> Void) throws {
            var t: [Double] = []
            for _ in 0..<5 {
                let cb = ctx.queue.makeCommandBuffer()!
                let e = encoder(cb)
                for i in 0..<count { body(e, i) }
                e.ce.endEncoding()
                cb.commit()
                try finish(cb)
                t.append((cb.gpuEndTime - cb.gpuStartTime) * 1e3)
            }
            print(String(format: "  %@ x%d: %.2f ms per frame, %.1f us each", name.padding(toLength: 26, withPad: " ", startingAt: 0), count, median(t), median(t) / Double(count) * 1e3))
        }
        for m in modes.prefix(2) {
            let l = m.loop, B = l.sequences, bb = weights.backbone.config, dd = weights.depth.config
            print("kernels without projection weights, \(m.name):")
            let bw = l.backbone.work, dw = l.depth.work
            let block = RowBlock(rows: 1, positions: m.positions)
            try part("backbone rmsnorm", l, count: 2 * bb.layers + 1) { e, i in
                e.rmsnorm(ActView(l.backboneInput, ld: bb.hidden), rows: B, width: bb.hidden, weight: weights.backbone.layers[i % bb.layers].norm1, ActView(bw.normed, ld: bb.hidden), eps: bb.eps)
            }
            try part("backbone norm_rope_kv", l, count: bb.layers) { e, i in
                e.normRopeKV(ActView(bw.qkv, ld: bb.qkvWidth), block: block, heads: bb.heads, kvHeads: bb.kvHeads, headDim: bb.headDim, cache: l.backbone.cache.layers[i],
                             invFreq: l.backbone.invFreq, norm: .weight(weights.backbone.layers[i].qkNorm!, eps: bb.eps))
            }
            try part("backbone attention", l, count: bb.layers) { e, i in
                e.attentionCausal(ActView(bw.qkv, ld: bb.qkvWidth), block: block, heads: bb.heads, kvHeads: bb.kvHeads, cache: l.backbone.cache.layers[i], ActView(bw.attn, ld: bb.qWidth))
            }
            try part("depth rmsnorm", l, count: 15 * (2 * dd.layers + 1)) { e, i in
                e.rmsnorm(ActView(l.depthX, ld: dd.hidden), rows: B, width: dd.hidden, weight: weights.depth.layers[i % dd.layers].norm1, ActView(dw.normed, ld: dd.hidden), eps: dd.eps)
            }
            try part("depth attention", l, count: 15 * dd.layers) { e, i in
                e.attentionShort(ActView(dw.qkv, ld: dd.qkvWidth), sequences: B, rows: 1, pos0: 2 + i % 14, heads: dd.heads, kvHeads: dd.kvHeads, cache: l.depth.cache.layers[i % dd.layers],
                                 invFreq: l.depth.invFreq, ActView(dw.attn, ld: dd.qWidth))
            }
            try part("embedding lookup", l, count: 15) { e, i in
                e.embedCodes(table: weights.embedding, width: BreezeShape.embedWidth, vocab: BreezeShape.codebookVocab, codes: l.codes, codesOffset: 0, firstCodebook: i, count: 1, rows: B, ActView(l.stepInput, ld: BreezeShape.embedWidth))
            }
            try part("sampler", l, count: 16) { e, i in
                e.sample(i == 0 ? .backbone : .depth, logits: ActView(l.logits, ld: FrameLoop.logitsStride), guidance: m.settings.guidance, settings: SamplerSettings(),
                         uniforms: l.uniforms, uniformsOffset: 0, codes: l.codes, codesOffset: 0, slot: i, seen: l.seen, status: l.status)
            }
        }
        // Where a guided frame's time goes, dispatch by dispatch.
        if ProcessInfo.processInfo.environment["BREEZE_TRACE"] != nil {
            let m = modes[1], traced = 4
            start(m)
            var rows: [String: DispatchTrace.Row] = [:]
            for f in 0..<traced {
                let cb = ctx.queue.makeCommandBuffer()!
                guard let trace = try DispatchTrace(ctx: ctx, commandBuffer: cb, capacity: 2 * dispatches) else { break }
                let e = Encoder(k: kernels, trace: trace)
                m.loop.encodeFrame(e, frame: f, settings: m.settings)
                e.ce.endEncoding()
                cb.commit()
                try finish(cb)
                if f == 0 { continue }
                for r in try trace.rows() {
                    let key = "\(r.kernel) \(r.weightBytes)"
                    rows[key, default: DispatchTrace.Row(kernel: r.kernel, weightBytes: r.weightBytes)].count += r.count
                    rows[key]!.seconds += r.seconds
                }
            }
            print("guided frame traced, one pass per dispatch (rows scaled to the traced frame's span, which pass boundaries lengthen):")
            print(DispatchTrace.report(rows.values.sorted { $0.seconds > $1.seconds }, frames: traced - 1))
        }
        print(String(format: "empty-dispatch floor: %d dispatches take %.2f ms (%.2f us each)", dispatches, median(floors), median(floors) / Double(dispatches) * 1e3))
    }
}
