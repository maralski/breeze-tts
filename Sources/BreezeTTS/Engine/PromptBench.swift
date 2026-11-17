// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// Times what a request waits on before its first frame: the text encoder per segment length
/// (with its attention dispatches alone, for attribution), the backbone prefill, and
/// `Synthesizer.run` to the first frame's codes, plain and guided. Lengths take turns within
/// each round so they share machine noise.
public enum PromptBench {
    public static func run(directory: URL, lengths: [Int] = [60, 400], rounds: Int = 7) throws {
        let synth = try Synthesizer(directory: directory)
        try synth.ctx.warmUpPipelines()
        let c = synth.text.config, w = BreezeShape.embedWidth
        var rng = SplitMix64(seed: 7)
        func ids(_ n: Int) -> [Int32] { [2] + (1..<n).map { _ in Int32(3 + rng.next() % 200_000) } }
        let (loop, prefill) = try synth.loop(sequences: 2)

        func gpu(_ body: (Encoder) -> Void) throws -> Double {
            let cb = try synth.commit { body($0) }
            try Synthesizer.wait(cb)
            return (cb.gpuEndTime - cb.gpuStartTime) * 1e3
        }
        struct Row { var encoder: [Double] = [], attention: [Double] = [], prefill: [Double] = [], plain: [Double] = [], guided: [Double] = [], guidedPrefill: [Double] = [] }
        var rows = [Row](repeating: Row(), count: lengths.count)
        var nonFinite = 0, stoppedAtOnce = 0
        // The GEMM row cut: each length under each cut, alternating within a round.
        if let cuts = ProcessInfo.processInfo.environment["BREEZE_GEMM_CUTS"]?.split(separator: ",").compactMap({ Int($0) }) {
            let saved = Kernels.gemmMinRows
            defer { Kernels.gemmMinRows = saved }
            var times = [[Double]](repeating: [], count: lengths.count * cuts.count)
            for round in 0...rounds {
                for (i, n) in lengths.enumerated() {
                    let segment = ids(n)
                    try synth.embedding.gather(segment, into: synth.encoderInput.contents().bindMemory(to: Float.self, capacity: n * c.hidden))
                    let layout = PromptLayout(BreezePrompt(positive: .init(segments: [.text(segment)]), guidance: nil))
                    for (j, cut) in (round % 2 == 0 ? Array(cuts.enumerated()) : cuts.enumerated().reversed()) {
                        Kernels.gemmMinRows = cut
                        let t = try gpu { e in
                            synth.text.encode(e, x: ActView(synth.encoderInput, ld: c.hidden), rows: n, out: synth.inputRows(0))
                            synth.encodePrefill(e, layout, loop: loop, prefill: prefill)
                        }
                        if round > 0 { times[i * cuts.count + j].append(t) }
                    }
                }
            }
            for (i, n) in lengths.enumerated() {
                print("\(n) tokens, encoder + prefill gpu ms by GEMM cut: " + cuts.enumerated().map { j, cut in "\(cut): \(FrameBench.span(times[i * cuts.count + j]))" }.joined(separator: ", "))
            }
            return
        }
        for round in 0...rounds {
            for (i, n) in (round % 2 == 0 ? Array(lengths.enumerated()) : lengths.enumerated().reversed()) {
                let segment = ids(n)
                let x = ActView(synth.encoderInput, ld: c.hidden)
                try synth.embedding.gather(segment, into: synth.encoderInput.contents().bindMemory(to: Float.self, capacity: n * c.hidden))
                let enc = try gpu { synth.text.encode($0, x: x, rows: n, out: synth.inputRows(0)) }
                nonFinite += synth.inputs[0].floats(count: n * w).filter { !$0.isFinite }.count
                let qkv = ActView(synth.text.work.qkv, ld: c.qkvWidth), attn = ActView(synth.text.work.attn, ld: c.qWidth)
                let att = try gpu { e in
                    for l in 0..<c.layers {
                        e.attentionWindow(qkv, rows: n, total: n, window: c.fullAttentionLayers.contains(l) ? nil : c.window,
                                          keys: synth.text.work.keys.k, values: synth.text.work.keys.v, scale: c.attentionScale, attn)
                    }
                }
                let layout = PromptLayout(BreezePrompt(positive: .init(segments: [.text(segment)]), guidance: nil))
                let pre = try gpu { synth.encodePrefill($0, layout, loop: loop, prefill: prefill) }
                var plain = BreezeSynthesisOptions()
                plain.maxFrames = 1
                let p = BreezePrompt(positive: .init(segments: [.text(segment)]), guidance: nil)
                let r1 = try synth.run(p, options: plain) { _ in }
                let g = BreezePrompt(positive: .init(segments: [.text(segment)]),
                                     guidance: .init(scale: 4, negative: .init(segments: [.text(Array(segment.prefix(n / 2)))])))
                let r2 = try synth.run(g, options: plain) { _ in }
                // Random tokens can draw end of speech at once: no first frame to time.
                stoppedAtOnce += [r1, r2].filter { $0.frameSeconds.isEmpty }.count
                if round > 0 {
                    rows[i].encoder.append(enc)
                    rows[i].attention.append(att)
                    rows[i].prefill.append(pre)
                    rows[i].plain.append((r1.prefillSeconds + (r1.frameSeconds.first ?? .nan)) * 1e3)
                    rows[i].guided.append((r2.prefillSeconds + (r2.frameSeconds.first ?? .nan)) * 1e3)
                    rows[i].guidedPrefill.append(r2.prefillSeconds * 1e3)
                }
            }
        }
        print("median [min-max] of \(rounds) interleaved rounds after one warm-up round; GPU times from command-buffer stamps, first frame is wall clock")
        for (i, n) in lengths.enumerated() {
            let r = rows[i]
            print(String(format: "%4d tokens: encoder gpu %@ ms (attention alone %@ ms), backbone prefill gpu %@ ms", n, FrameBench.span(r.encoder), FrameBench.span(r.attention), FrameBench.span(r.prefill)))
            print(String(format: "             first frame wall: plain %@ ms, guided (negative %d tokens) %@ ms of which prompt %@ ms", FrameBench.span(r.plain), n / 2, FrameBench.span(r.guided), FrameBench.span(r.guidedPrefill)))
        }
        print("work check: non-finite projected values \(nonFinite), runs that stopped before a first frame \(stoppedAtOnce)")
    }
}
