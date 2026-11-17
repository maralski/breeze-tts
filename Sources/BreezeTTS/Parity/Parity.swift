// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// One compared tensor. `cosine` is the worst over its rows (a frame, a branch, a codebook);
/// `referenceCosine` is the same measure between the reference's own fp32 and bf16 runs, which
/// shows how far the reference itself moves under rounding.
public struct ParityLine: Sendable {
    public let name: String
    public let shape: [Int]
    public let cosine: Double
    public let maxAbs: Float
    public let referenceCosine: Double?
    public let threshold: Double
    public var passed: Bool { cosine >= threshold }

    public var description: String {
        let ref = referenceCosine.map { String(format: "%.8f", $0) } ?? "-"
        return String(format: "%@ %@ %@ cos %.8f  max|d| %.3g  ref(bf16) %@%@", passed ? "PASS" : "FAIL",
                      name.padding(toLength: 40, withPad: " ", startingAt: 0), "\(shape)".padding(toLength: 18, withPad: " ", startingAt: 0),
                      cosine, maxAbs, ref, "")
    }
}

/// A token stream compared draw by draw.
public struct StreamLine: Sendable {
    public let name: String
    public let frames: Int
    public let referenceFrames: Int
    /// Draws whose token differs from the reference's.
    public let mismatches: Int
    /// The first mismatching draw: frame, codebook, and a note on how close a call it was.
    public let first: (frame: Int, codebook: Int, note: String)?
    /// Whether a mismatch is allowed as a numeric near-tie (decided by the caller's rule).
    public let passed: Bool

    public var description: String {
        let where_ = first.map { "first at frame \($0.frame) codebook \($0.codebook): \($0.note)" } ?? "identical"
        return "\(passed ? "PASS" : "FAIL") \(name.padding(toLength: 40, withPad: " ", startingAt: 0)) \(frames) frames (reference \(referenceFrames)), \(mismatches) differing draws; \(where_)"
    }
}

public struct ParityReport: Sendable {
    public let caseName: String
    public var tensors: [ParityLine] = []
    public var streams: [StreamLine] = []
    public var notes: [String] = []
    public var passed: Bool { tensors.allSatisfy(\.passed) && streams.allSatisfy(\.passed) }
    public var worstCosine: Double { tensors.map(\.cosine).min() ?? 1 }

    public var description: String {
        var out = ["case \(caseName)"] + tensors.map(\.description) + streams.map(\.description) + notes.map { "note \($0)" }
        let failed = tensors.filter { !$0.passed }.count + streams.filter { !$0.passed }.count
        out.append(String(format: "summary %@: %d tensors, worst cosine %.6f, %d streams, %d failed", caseName, tensors.count, worstCosine, streams.count, failed))
        return out.joined(separator: "\n")
    }
}

/// Compares the port with the reference goldens stage by stage, each stage fed the reference's
/// own inputs: the text encoder its token ids, the backbone prefill the reference's
/// `inputs_embeds`, the frame stream the reference's codes (teacher forcing). The free-running
/// streams then go through `Synthesizer.run` end to end.
public final class BreezeParity {
    let synth: Synthesizer
    let tokenizer: BreezeTokenizer
    public static let threshold = 0.9999

    public init(weights: URL) throws {
        synth = try Synthesizer(directory: weights)
        tokenizer = try BreezeTokenizer(tokenizerJSON: weights.appendingPathComponent("tokenizer.json"))
    }

    // MARK: comparison

    static func cosine(_ a: ArraySlice<Float>, _ b: ArraySlice<Float>) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) {
            dot += Double(x) * Double(y)
            na += Double(x) * Double(x)
            nb += Double(y) * Double(y)
        }
        if na == 0 && nb == 0 { return 1 }
        return dot / (na * nb).squareRoot()
    }

    /// Worst row cosine and max abs error of `got` against `want`, rows of `width`.
    static func compare(_ got: [Float], _ want: [Float], width: Int) -> (cosine: Double, maxAbs: Float) {
        precondition(got.count == want.count && want.count % width == 0, "compare: \(got.count) vs \(want.count) at width \(width)")
        var worst = 1.0
        var maxAbs: Float = 0
        for r in 0..<(want.count / width) {
            let range = (r * width)..<((r + 1) * width)
            let c = cosine(got[range], want[range])
            worst = min(worst, c.isNaN ? -1 : c)
        }
        for (x, y) in zip(got, want) { maxAbs = max(maxAbs, abs(x - y)) }
        if got.contains(where: { $0.isNaN }) { worst = -1 }
        return (worst, maxAbs)
    }

    // MARK: GPU capture

    final class Captures {
        let ctx: MetalContext
        private(set) var buffers: [String: MTLBuffer] = [:]
        init(ctx: MetalContext) { self.ctx = ctx }

        /// A capture buffer of at least `floats`, made on first use.
        func buffer(_ name: String, floats: Int) -> MTLBuffer {
            if let b = buffers[name], b.length >= floats * 4 { return b }
            let b = try! ctx.makeBuffer(bytes: floats * 4, label: "capture.\(name)")
            buffers[name] = b
            return b
        }

        func copy(_ e: Encoder, _ name: String, from: ActView, rows: Int, width: Int, at row: Int = 0) {
            let b = buffer(name, floats: (row + rows) * width)
            for r in 0..<rows { e.copy(from.row(r), count: width, to: ActView(b, offset: (row + r) * width * 4, ld: width)) }
        }

        func read(_ name: String, count: Int, from row: Int = 0, width: Int = 0) -> [Float] {
            buffers[name]!.floats(count: count, offset: row * width * 4)
        }
    }

    // MARK: run

    public func run(caseDirectory: URL) throws -> ParityReport {
        let golden = try GoldenCase(directory: caseDirectory)
        var report = ParityReport(caseName: golden.name)
        func add(_ name: String, _ got: [Float], _ want: GoldenCase.Tensor, width: Int, bf16: String? = nil) throws {
            let (c, m) = Self.compare(got, want.values, width: width)
            var ref: Double?
            if let other = try golden.tensor(bf16 ?? name, precision: "bf16"), other.values.count == want.values.count {
                ref = Self.compare(other.values, want.values, width: width).cosine
            }
            report.tensors.append(ParityLine(name: name, shape: want.shape, cosine: c, maxAbs: m, referenceCosine: ref, threshold: Self.threshold))
        }

        // Front end: the prompt the port builds from the case's request against the reference's.
        let prompt = try golden.referencePrompt()
        if let text = golden.text {
            let built = try BreezePrompt(fragment: text, of: try golden.request(), guidance: golden.guidance, tokenizer: tokenizer)
            let builtOK = built == prompt
            report.streams.append(StreamLine(name: "front_end_prompt", frames: built.positive.count, referenceFrames: prompt.positive.count,
                                             mismatches: builtOK ? 0 : 1, first: builtOK ? nil : (0, 0, "built prompt differs from the manifest's input_ids and segments"), passed: builtOK))
        }

        let layout = PromptLayout(prompt)
        let (loop, prefill) = try synth.loop(sequences: layout.branches.count)
        let caps = Captures(ctx: synth.ctx)
        let hidden = synth.text.config.hidden, w = BreezeShape.embedWidth
        let textTaps = [0, 4, 5, 25]

        // 1, 2: text encoder per segment and prompt assembly.
        try synth.upload(layout)
        var segments: [(branch: String, index: Int, text: PromptLayout.Text)] = []
        for (b, branch) in layout.branches.enumerated() {
            for (i, t) in branch.texts.enumerated() where t.encoderRow != nil { segments.append((golden.branches[b], i, t)) }
        }
        try Synthesizer.wait(try synth.commit { e in
            synth.encodePrompt(e, layout, afterLayer: { s, layer in
                guard textTaps.contains(layer) else { return }
                let t = segments[s].text
                caps.copy(e, "text\(s)_layer\(layer)", from: ActView(self.synth.encoderInput, offset: t.encoderRow! * hidden * 4, ld: hidden), rows: t.ids.count, width: hidden)
            }, afterNorm: { s, normed in
                caps.copy(e, "text\(s)_final_norm", from: normed, rows: segments[s].text.ids.count, width: hidden)
            })
        })
        for (s, seg) in segments.enumerated() {
            let L = seg.text.ids.count, prefix = "text_\(seg.branch)_seg\(seg.index)_"
            var embed = [Float](repeating: 0, count: L * hidden)
            try embed.withUnsafeMutableBufferPointer { try synth.embedding.gather(seg.text.ids, into: $0.baseAddress!) }
            try add(prefix + "embed", embed, try golden.require(prefix + "embed"), width: hidden)
            for layer in textTaps {
                try add(prefix + "layer\(layer)", caps.read("text\(s)_layer\(layer)", count: L * hidden), try golden.require(prefix + "layer\(layer)"), width: hidden)
            }
            try add(prefix + "final_norm", caps.read("text\(s)_final_norm", count: L * hidden), try golden.require(prefix + "final_norm"), width: hidden)
            let b = golden.branches.firstIndex(of: seg.branch)!
            try add(prefix + "proj", synth.inputs[b].floats(count: L * w, offset: seg.text.row * w * 4), try golden.require(prefix + "proj"), width: w)
        }
        // A text-encoder-only case (long_text) stops here.
        guard try golden.tensor("prefill_cond_inputs_embeds") != nil else { return report }
        for (b, branch) in layout.branches.enumerated() {
            let name = "prefill_\(golden.branches[b])_inputs_embeds"
            try add(name, synth.inputs[b].floats(count: branch.rows * w), try golden.require(name), width: w)
        }

        // 3: backbone prefill from the reference's inputs_embeds.
        for b in layout.branches.indices {
            let want = try golden.require("prefill_\(golden.branches[b])_inputs_embeds")
            synth.inputs[b].write(want.values)
        }
        try Synthesizer.wait(try synth.commit { e in
            synth.encodePrefill(e, layout, loop: loop, prefill: prefill, afterLayer: { b, layer in
                if layer == 0 { caps.copy(e, "prefill\(b)_layer0", from: self.synth.inputRows(b), rows: layout.branches[b].rows, width: w) }
            })
            let bb = synth.frameWeights.backbone.config
            for (b, branch) in layout.branches.enumerated() {
                let norm = caps.buffer("prefill\(b)_final_norm", floats: (branch.rows + 8) * w)
                e.rmsnorm(synth.inputRows(b), rows: branch.rows, width: w, weight: synth.frameWeights.backbone.finalNorm, ActView(norm, ld: w), eps: bb.eps)
                let logits = caps.buffer("prefill\(b)_logits", floats: 8 * FrameLoop.logitsStride)
                e.linear(ActView(loop.depthInput, offset: b * 2 * w * 4, ld: 2 * w), rows: 1, synth.frameWeights.lmHead, ActView(logits, ld: FrameLoop.logitsStride))
            }
        })
        for (b, branch) in layout.branches.enumerated() {
            let p = "prefill_\(golden.branches[b])_"
            try add(p + "layer0", caps.read("prefill\(b)_layer0", count: branch.rows * w), try golden.require(p + "layer0"), width: w)
            try add(p + "layer27", synth.inputs[b].floats(count: branch.rows * w), try golden.require(p + "layer27"), width: w)
            try add(p + "final_norm", caps.read("prefill\(b)_final_norm", count: branch.rows * w), try golden.require(p + "final_norm"), width: w)
            try add(p + "last_logits", caps.read("prefill\(b)_logits", count: BreezeShape.backboneVocab), try golden.require(p + "last_logits"), width: BreezeShape.backboneVocab)
        }

        // 4: the frame streams, teacher-forced, then free-running.
        for mode in ["greedy", "sampled"] {
            let picks = try teacherForced(golden, mode: mode, layout: layout, loop: loop, prefill: prefill, add: add)
            report.streams.append(picks)
            let (line, note) = try freeRunning(golden, mode: mode, prompt: prompt)
            report.streams.append(line)
            report.notes.append(note)
        }
        return report
    }

    struct Draws {
        /// fp32 [frames, B, 2052] lm_head rows, [frames, 15, B, 2051] codebook-head rows.
        var backbone: [Float] = []
        var depth: [Float] = []
    }
    private var teacherDraws: [String: Draws] = [:]

    private func settings(_ golden: GoldenCase, mode: String) -> BreezeSynthesisOptions {
        var o = BreezeSynthesisOptions()
        o.greedy = mode == "greedy"
        return o
    }

    private func uniforms(_ golden: GoldenCase, mode: String, frames: Int) throws -> [[Float]] {
        guard mode == "sampled" else { return [[Float]](repeating: [Float](repeating: 0, count: 16), count: frames) }
        let ub = try golden.require("sampled_u_backbone").values, ud = try golden.require("sampled_u_depth").values
        return (0..<frames).map { f in
            [f < ub.count ? ub[f] : 0.5] + (0..<15).map { q in f * 15 + q < ud.count ? ud[f * 15 + q] : 0.5 }
        }
    }

    private func teacherForced(_ golden: GoldenCase, mode: String, layout: PromptLayout, loop: FrameLoop, prefill: DecoderForward,
                               add: (String, [Float], GoldenCase.Tensor, Int, String?) throws -> Void) throws -> StreamLine {
        let B = layout.branches.count, w = BreezeShape.embedWidth, V = FrameLoop.logitsStride, books = BreezeShape.codebooks
        let codes = try golden.ints("\(mode)_codes")
        let n = codes.shape[0], steps = n + 1
        let caps = Captures(ctx: synth.ctx)
        // Prefill again from the reference's inputs (the previous stage left them in place only
        // until its residual stream overwrote them).
        for b in layout.branches.indices { synth.inputs[b].write(try golden.require("prefill_\(golden.branches[b])_inputs_embeds").values) }
        try Synthesizer.wait(try synth.commit { synth.encodePrefill($0, layout, loop: loop, prefill: prefill) })
        loop.begin(positions: layout.branches.map(\.rows))
        let teacher = try synth.ctx.makeBuffer(bytes: steps * books * 4, label: "teacher")
        let tp = teacher.contents().bindMemory(to: UInt32.self, capacity: steps * books)
        for i in 0..<(n * books) { tp[i] = UInt32(codes.values[i]) }
        tp[n * books] = UInt32(BreezeShape.endOfSpeech)
        loop.teacher = teacher
        defer { loop.teacher = nil; loop.tap = nil }
        let u = try uniforms(golden, mode: mode, frames: steps)
        var own: [[Int]] = []
        let opts = settings(golden, mode: mode).frameSettings(guidance: B == 2 ? golden.guidance : nil)
        let seen = loop.seen.contents().bindMemory(to: UInt32.self, capacity: BreezeShape.backboneVocab)
        // Captures grow by replacement, so size them for the whole stream up front.
        _ = caps.buffer("hidden", floats: steps * B * w)
        _ = caps.buffer("backbone", floats: steps * B * V)
        _ = caps.buffer("depth", floats: steps * 15 * B * V)
        for f in 0..<steps {
            loop.tap = { e, tap in
                switch tap {
                case .backboneHidden: caps.copy(e, "hidden", from: ActView(loop.depthInput, ld: 2 * w), rows: B, width: w, at: f * B)
                case .backboneLogits: caps.copy(e, "backbone", from: ActView(loop.logits, ld: V), rows: B, width: V, at: f * B)
                case .depthLogits(let q): caps.copy(e, "depth", from: ActView(loop.logits, ld: V), rows: B, width: V, at: (f * 15 + q - 1) * B)
                }
            }
            // The reference's history, whatever this port drew.
            memset(loop.seen.contents(), 0, loop.seen.length)
            for g in 0..<f { seen[Int(codes.values[g * books])] = 1 }
            loop.setUniforms(frame: f, u[f])
            try Synthesizer.wait(try synth.commit { e in
                if f == 0 { loop.encodeCodes(e, frame: 0, settings: opts) } else { loop.encodeFrame(e, frame: f, settings: opts) }
            })
            own.append(loop.frameCodes(f))
        }
        let hidden = caps.read("hidden", count: n * B * w)
        let backbone = caps.read("backbone", count: steps * B * V)
        let depthRaw = caps.read("depth", count: n * 15 * B * V)
        func trim(_ rows: [Float], from stride: Int, to width: Int) -> [Float] {
            (0..<(rows.count / stride)).flatMap { rows[($0 * stride)..<($0 * stride + width)] }
        }
        let depth = trim(depthRaw, from: V, to: BreezeShape.codebookVocab)
        let p = "\(mode)_"
        try add("tf_" + p + "backbone_hidden", hidden, try golden.require(p + "backbone_hidden"), w, p + "backbone_hidden")
        try add("tf_" + p + "backbone_logits", backbone, try golden.require(p + "backbone_logits"), BreezeShape.backboneVocab, p + "backbone_logits")
        try add("tf_" + p + "depth_logits", depth, try golden.require(p + "depth_logits"), BreezeShape.codebookVocab, p + "depth_logits")
        let s = golden.guidance
        func combine(_ rows: [Float], width: Int) -> [Float] {
            guard B == 2 else { return rows }
            return (0..<(rows.count / (2 * width))).flatMap { r -> [Float] in
                (0..<width).map { i in
                    let c = rows[2 * r * width + i], un = rows[(2 * r + 1) * width + i]
                    let diff = c - un
                    let scaled = s * diff
                    return un + scaled
                }
            }
        }
        let backboneCFG = combine(backbone, width: BreezeShape.backboneVocab), depthCFG = combine(depth, width: BreezeShape.codebookVocab)
        try add("tf_" + p + "backbone_logits_cfg", backboneCFG, try golden.require(p + "backbone_logits_cfg"), BreezeShape.backboneVocab, p + "backbone_logits_cfg")
        try add("tf_" + p + "depth_logits_cfg", depthCFG, try golden.require(p + "depth_logits_cfg"), BreezeShape.codebookVocab, p + "depth_logits_cfg")
        teacherDraws[p] = Draws(backbone: backboneCFG, depth: depthCFG)

        // The port's own picks under teacher forcing against the reference's.
        let refBackbone = try golden.ints(p + "backbone_token_own").values, refDepth = try golden.ints(p + "depth_token_own").values
        var mismatches: [(Int, Int)] = []
        for f in 0..<steps {
            if own[f][0] != Int(refBackbone[f]) { mismatches.append((f, 0)) }
            if f < n { for q in 1..<books where own[f][q] != Int(refDepth[f * 15 + q - 1]) { mismatches.append((f, q)) } }
        }
        let notes = try mismatches.map { try margin(golden, mode: mode, frame: $0.0, codebook: $0.1) }
        return (StreamLine(name: "tf_\(mode)_own_picks", frames: steps, referenceFrames: steps, mismatches: mismatches.count,
                                         first: mismatches.first.map { ($0.0, $0.1, notes[0].note) }, passed: notes.allSatisfy(\.nearTie)))
    }

    /// How close a call draw (frame, codebook) was, on the reference's logits and on this port's
    /// teacher-forced ones: the top-two gap for greedy, the uniform's distance to the nearest
    /// cumulative boundary for sampling. A near-tie is a gap within the error between the two.
    private func margin(_ golden: GoldenCase, mode: String, frame f: Int, codebook q: Int) throws -> (note: String, nearTie: Bool) {
        let p = "\(mode)_"
        guard let draws = teacherDraws[p] else { return ("no teacher-forced logits", false) }
        let kind: SamplerKind = q == 0 ? .backbone : .depth
        let width = q == 0 ? BreezeShape.backboneVocab : BreezeShape.codebookVocab
        let refAll = try golden.require(p + (q == 0 ? "backbone_logits_cfg" : "depth_logits_cfg")).values
        let row = q == 0 ? f : f * 15 + q - 1
        guard (row + 1) * width <= refAll.count else { return ("past the reference's stream", false) }
        let ref = Array(refAll[(row * width)..<((row + 1) * width)])
        let port = Array((q == 0 ? draws.backbone : draws.depth)[(row * width)..<((row + 1) * width)])
        let codes = try golden.ints(p + "codes")
        let seen = Set((0..<min(f, codes.shape[0])).map { Int(codes.values[$0 * 16]) })
        var settings = SamplerSettings()
        settings.greedy = mode == "greedy"
        let err = zip(ref, port).map { abs($0 - $1) }.max() ?? 0
        if settings.greedy {
            func gap(_ x: [Float]) -> (Int, Float) {
                let s = SamplingRule.scores(kind, cond: x, uncond: nil, guidance: 1, seen: kind == .backbone ? seen : [], settings: settings)
                let order = s.indices.sorted { s[$0] > s[$1] }
                return (order[0], s[order[0]] - s[order[1]])
            }
            let (rt, rg) = gap(ref), (pt, pg) = gap(port)
            return (String(format: "reference picks %d by %.3g, port %d by %.3g, max|d| logits %.3g", rt, rg, pt, pg, err), rg <= 2 * err)
        }
        let uAll = try golden.require(q == 0 ? "sampled_u_backbone" : "sampled_u_depth").values
        let u = uAll[q == 0 ? f : f * 15 + q - 1]
        func boundary(_ x: [Float]) -> (Int, Float) {
            let d = SamplingRule.draw(kind, cond: x, seen: kind == .backbone ? seen : [], settings: settings, uniform: u)
            let total = d.cumulative.last!
            let lo = d.token > 0 ? d.cumulative[d.token - 1] : 0, hi = d.cumulative[d.token]
            return (d.token, min(d.target - lo, hi - d.target) / total)
        }
        let (rt, rg) = boundary(ref), (pt, pg) = boundary(port)
        return (String(format: "reference draws %d at %.3g of the mass from a boundary, port %d at %.3g, max|d| logits %.3g", rt, rg, pt, pg, err), min(rg, pg) <= 3 * err)
    }

    private func freeRunning(_ golden: GoldenCase, mode: String, prompt: BreezePrompt) throws -> (StreamLine, String) {
        let codes = try golden.ints("\(mode)_codes")
        let n = codes.shape[0]
        var options = settings(golden, mode: mode)
        options.maxFrames = golden.frameCap
        options.uniforms = .recorded(try uniforms(golden, mode: mode, frames: golden.frameCap + 1))
        var got: [[Int]] = []
        let r = try synth.run(prompt, options: options) { got.append($0) }
        var first: (Int, Int)?
        for f in 0...n where first == nil {
            let want: [Int]? = f < n ? (0..<16).map { Int(codes.values[f * 16 + $0]) } : nil
            let have: [Int]? = f < got.count ? got[f] : nil
            if want == have { continue }
            if let want, let have { first = (f, want.indices.first { want[$0] != have[$0] }!) } else { first = (f, 0) }
        }
        var note = "", passed = true
        if let first {
            let m = try margin(golden, mode: mode, frame: first.0, codebook: first.1)
            (note, passed) = (m.note, m.nearTie)
        }
        let line = (StreamLine(name: "free_\(mode)_codes", frames: got.count, referenceFrames: n, mismatches: first == nil ? 0 : 1,
                                         first: first.map { ($0.0, $0.1, note) }, passed: passed && (first != nil || r.stoppedOnEndOfSpeech)))
        return (line, String(format: "free_%@: %d frames, stopped on end of speech %@, prefill %.1f ms, median frame %.1f ms", mode, r.frames,
                                   r.stoppedOnEndOfSpeech ? "yes" : "no", r.prefillSeconds * 1e3, (r.frameSeconds.sorted().dropFirst(r.frameSeconds.count / 2).first ?? 0) * 1e3))
    }
}
