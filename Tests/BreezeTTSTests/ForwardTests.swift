// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import BreezeTTS

/// The decoder stack and the frame schedule against a scalar forward written from the model
/// description: one row at a time, attention over every earlier row.
final class ForwardTests: GPUTestCase {
    struct HostLayer {
        var norm1, qkv, qkNorm, o, norm2, gate, up, down: [Float]
    }

    /// Random weights for `config`, as the GPU tensors and the bf16-rounded host copy.
    func makeDecoder(_ c: DecoderConfig, rng: inout RNG) throws -> (DecoderWeights, [HostLayer], [Float]) {
        func mat(_ n: Int, _ k: Int) throws -> (Linear, [Float]) {
            let (t, f) = try bf16Tensor(rng.floats(n * k, scale: 1.5 / Float(k).squareRoot()))
            return (Linear(w: t, bias: nil, n: n, k: k), f)
        }
        func vec(_ n: Int) throws -> (GPUTensor, [Float]) { try bf16Tensor((0..<n).map { _ in 1 + rng.next() * 0.3 }) }
        var gpu: [DecoderLayerWeights] = [], host: [HostLayer] = []
        for _ in 0..<c.layers {
            let n1 = try vec(c.hidden), qkv = try mat(c.qkvWidth, c.hidden), o = try mat(c.hidden, c.qWidth), n2 = try vec(c.hidden)
            let qk = c.qkNorm ? try vec(2 * c.headDim) : nil
            let gate = try mat(c.intermediate, c.hidden), up = try mat(c.intermediate, c.hidden), down = try mat(c.hidden, c.intermediate)
            gpu.append(DecoderLayerWeights(norm1: n1.0, qkv: qkv.0, qkNorm: qk?.0, o: o.0, norm2: n2.0, gate: gate.0, up: up.0, down: down.0))
            host.append(HostLayer(norm1: n1.1, qkv: qkv.1, qkNorm: qk?.1 ?? [], o: o.1, norm2: n2.1, gate: gate.1, up: up.1, down: down.1))
        }
        let fin = try vec(c.hidden)
        return (DecoderWeights(config: c, layers: gpu, finalNorm: fin.0), host, fin.1)
    }

    static func matvec(_ w: [Float], _ x: [Float], n: Int) -> [Float] {
        let k = x.count
        return w.withUnsafeBufferPointer { wp in
            x.withUnsafeBufferPointer { xp in
                (0..<n).map { r in
                    var s: Float = 0
                    let row = wp.baseAddress! + r * k
                    for i in 0..<k { s += row[i] * xp[i] }
                    return s
                }
            }
        }
    }

    static func rmsnorm(_ x: [Float], _ w: [Float], eps: Float) -> [Float] {
        let s = 1 / (dot(x[...], x[...]) / Float(x.count) + eps).squareRoot()
        return zip(x, w).map { $0 * s * $1 }
    }

    /// One sequence's reference forward with its growing key and value history.
    final class HostSequence {
        let c: DecoderConfig
        let layers: [HostLayer]
        let finalNorm: [Float]
        var keys: [[[[Float]]]], values: [[[[Float]]]]
        var position = 0

        init(_ c: DecoderConfig, _ layers: [HostLayer], _ finalNorm: [Float]) {
            (self.c, self.layers, self.finalNorm) = (c, layers, finalNorm)
            keys = [[[[Float]]]](repeating: [[[Float]]](repeating: [], count: c.kvHeads), count: c.layers)
            values = keys
        }

        /// Runs one new row and returns its final-normed hidden state.
        func step(_ input: [Float]) -> [Float] {
            var h = input
            let D = c.headDim, G = c.heads / c.kvHeads
            for (i, l) in layers.enumerated() {
                let qkv = ForwardTests.matvec(l.qkv, ForwardTests.rmsnorm(h, l.norm1, eps: c.eps), n: c.qkvWidth)
                func head(_ slot: Int, normWith w: ArraySlice<Float>?, rotate: Bool) -> [Float] {
                    var v = Array(qkv[(slot * D)..<((slot + 1) * D)])
                    if let w { v = ForwardTests.rmsnorm(v, Array(w), eps: c.eps) }
                    return rotate ? GPUTestCase.rotateHalf(v, pos: position, invFreq: c.rope.values) : v
                }
                for kh in 0..<c.kvHeads {
                    keys[i][kh].append(head(c.heads + kh, normWith: c.qkNorm ? l.qkNorm[D...] : nil, rotate: true))
                    values[i][kh].append(head(c.heads + c.kvHeads + kh, normWith: nil, rotate: false))
                }
                var attn: [Float] = []
                for qh in 0..<c.heads {
                    let q = head(qh, normWith: c.qkNorm ? l.qkNorm[..<D] : nil, rotate: true)
                    let scores = keys[i][qh / G].map { GPUTestCase.dot(q[...], $0[...]) * c.attentionScale }
                    attn += GPUTestCase.attend(scores, values[i][qh / G])
                }
                h = zip(h, ForwardTests.matvec(l.o, attn, n: c.hidden)).map(+)
                let n2 = ForwardTests.rmsnorm(h, l.norm2, eps: c.eps)
                let gate = ForwardTests.matvec(l.gate, n2, n: c.intermediate), up = ForwardTests.matvec(l.up, n2, n: c.intermediate)
                let act = zip(gate, up).map { LinearTests.silu($0) * $1 }
                h = zip(h, ForwardTests.matvec(l.down, act, n: c.hidden)).map(+)
            }
            position += 1
            return ForwardTests.rmsnorm(h, finalNorm, eps: c.eps)
        }
    }

    /// Runs `schedule` (rows per call; a call names its sequences) through DecoderForward and
    /// through the reference, comparing the final hidden state of each call's last row.
    func stackCase(_ c: DecoderConfig, schedule: [(sequences: Range<Int>, rows: Int)], total: Int, seed: UInt64) throws {
        var rng = RNG(state: seed)
        let (weights, hostLayers, fin) = try makeDecoder(c, rng: &rng)
        let cache = try KVCache(ctx: ctx, config: c, sequences: total, label: "kv")
        let forward = try DecoderForward(ctx: ctx, weights: weights, cache: cache, work: try DecoderWorkspace(ctx: ctx, config: c, maxRows: 16, label: "w"))
        let hosts = (0..<total).map { _ in HostSequence(c, hostLayers, fin) }
        for (call, step) in schedule.enumerated() {
            let inputs = step.sequences.map { _ in (0..<step.rows).map { _ in rng.floats(c.hidden) } }
            let xb = try f32Buffer(inputs.flatMap { $0.flatMap { $0 } } + [Float](repeating: .nan, count: 8 * c.hidden))
            let out = try nanBuffer(floats: step.sequences.count * c.hidden)
            try run(concurrent: call % 2 == 1) { e in
                forward.encode(e, x: ActView(xb, ld: c.hidden), rows: step.rows, sequences: step.sequences, out: ActView(out, ld: c.hidden))
            }
            var want: [Float] = []
            for (i, s) in step.sequences.enumerated() {
                var last: [Float] = []
                for row in inputs[i] { last = hosts[s].step(row) }
                want += last
            }
            assertClose(out.floats(count: want.count), want, atol: 2e-4, rtol: 2e-4, "\(c.attention) call \(call)")
        }
        XCTAssertEqual(cache.lengths, hosts.map(\.position))
    }

    /// The backbone's shape in small: q/k norm, two sequences prefilled apart to different
    /// lengths, then decoded together.
    func testCachedDecoderStack() throws {
        var c = DecoderConfig.backbone
        (c.hidden, c.heads, c.kvHeads, c.intermediate, c.layers, c.capacity) = (256, 2, 1, 512, 2, 64)
        try stackCase(c, schedule: [(0..<1, 5), (1..<2, 3), (0..<2, 1), (0..<2, 1), (0..<2, 2)], total: 2, seed: 41)
    }

    /// The depth decoder's shape in small: two rows, then single rows, two sequences throughout.
    func testShortDecoderStack() throws {
        var c = DecoderConfig.depth
        (c.hidden, c.heads, c.kvHeads, c.intermediate, c.layers) = (512, 4, 1, 1024, 2)
        try stackCase(c, schedule: [(0..<2, 2)] + [(sequences: Range<Int>, rows: Int)](repeating: (0..<2, 1), count: 5), total: 2, seed: 42)
        try stackCase(c, schedule: [(0..<1, 2), (0..<1, 1)], total: 1, seed: 43)
    }

    /// Whole frames at the model's real widths, one layer per stack, greedy: the 16 codes of
    /// each frame and the next backbone input must match a reference that follows the frame
    /// description step by step. Guided, the two sequences start at different positions.
    func frameCase(guidance: Float?, frames: Int, seed: UInt64) throws {
        var rng = RNG(state: seed)
        var bb = DecoderConfig.backbone, dd = DecoderConfig.depth
        (bb.layers, bb.capacity, dd.layers) = (1, 32, 1)
        let (bw, bHost, bFin) = try makeDecoder(bb, rng: &rng)
        let (dw, dHost, dFin) = try makeDecoder(dd, rng: &rng)
        let w = BreezeShape.embedWidth, vocab = BreezeShape.codebookVocab, books = BreezeShape.codebooks
        func mat(_ n: Int, _ k: Int, _ scale: Float) throws -> (Linear, [Float]) {
            let (t, f) = try bf16Tensor(rng.floats(n * k, scale: scale))
            return (Linear(w: t, bias: nil, n: n, k: k), f)
        }
        let lm = try mat(BreezeShape.backboneVocab, w, 0.05), projector = try mat(dd.hidden, w, 0.03)
        let heads = try (0..<(books - 1)).map { _ in try mat(vocab, dd.hidden, 0.05) }
        let table = try mat(books * vocab, w, 0.5)
        let weights = FrameWeights(backbone: bw, lmHead: lm.0, depth: dw, depthProjector: projector.0, codebookHeads: heads.map(\.0), embedding: table.0.w)
        var settings = FrameSettings()
        settings.guidance = guidance
        settings.backbone.greedy = true
        settings.depth.greedy = true
        let B = settings.sequences
        let loop = try FrameLoop(ctx: ctx, weights: weights, sequences: B, maxFrames: frames)

        // Two prompt rows per sequence through the stack, the second sequence one row shorter.
        let prompt = (0..<B).map { s in (0..<(3 - s)).map { _ in rng.floats(w) } }
        let hosts = (0..<B).map { _ in HostSequence(bb, bHost, bFin) }
        // Prefill is a second forward over the loop's cache with a workspace of its own size.
        let prefill = try DecoderForward(ctx: ctx, weights: bw, cache: loop.backbone.cache, work: try DecoderWorkspace(ctx: ctx, config: bb, maxRows: 4, label: "prefill"))
        for s in 0..<B {
            let rows = Array(prompt[s].dropLast())
            let xb = try f32Buffer(rows.flatMap { $0 } + [Float](repeating: .nan, count: 8 * w)), sink = try nanBuffer(floats: w)
            try run { e in prefill.encode(e, x: ActView(xb, ld: w), rows: rows.count, sequences: s..<(s + 1), out: ActView(sink, ld: w)) }
            for row in rows { _ = hosts[s].step(row) }
        }
        loop.begin(positions: hosts.map(\.position))
        loop.backboneInput.write(prompt.flatMap { $0.last! })
        var inputs = prompt.map { $0.last! }
        var history = Set<Int>()
        for frame in 0..<frames {
            loop.setUniforms(frame: frame, [Float](repeating: 0.5, count: books))
            try run(concurrent: frame % 2 == 1) { e in loop.encodeFrame(e, frame: frame, settings: settings) }
            XCTAssertEqual(loop.statusWord, 0)

            func pick(_ kind: SamplerKind, _ rows: [[Float]]) -> Int {
                let s = kind == .backbone ? settings.backbone : settings.depth
                return SamplingRule.draw(kind, cond: rows[0], uncond: B == 2 ? rows[1] : nil, guidance: guidance ?? 1, seen: history, settings: s, uniform: 0.5).token
            }
            func embedding(_ code: Int, _ q: Int) -> [Float] { Array(table.1[((code + q * vocab) * w)..<((code + q * vocab + 1) * w)]) }
            let hidden = (0..<B).map { hosts[$0].step(inputs[$0]) }
            var want = [pick(.backbone, hidden.map { Self.matvec(lm.1, $0, n: BreezeShape.backboneVocab) })]
            history.insert(want[0])
            let depth = (0..<B).map { _ in HostSequence(dd, dHost, dFin) }
            for s in 0..<B { _ = depth[s].step(Self.matvec(projector.1, hidden[s], n: dd.hidden)) }
            for q in 1..<books {
                let x = Self.matvec(projector.1, embedding(want[q - 1], q - 1), n: dd.hidden)
                want.append(pick(.depth, (0..<B).map { Self.matvec(heads[q - 1].1, depth[$0].step(x), n: vocab) }))
            }
            XCTAssertEqual(loop.frameCodes(frame), want, "codes of frame \(frame), guidance \(String(describing: guidance))")
            var next = [Float](repeating: 0, count: w)
            for q in 0..<books { for (d, v) in embedding(want[q], q).enumerated() { next[d] += v } }
            inputs = [[Float]](repeating: next, count: B)
            XCTAssertEqual(loop.backboneInput.floats(count: B * w), inputs.flatMap { $0 }, "next backbone input after frame \(frame)")
        }
        XCTAssertEqual(loop.backbone.cache.lengths, hosts.map(\.position))
    }

    func testFrameMatchesReference() throws {
        try frameCase(guidance: nil, frames: 2, seed: 51)
        try frameCase(guidance: 4, frames: 2, seed: 52)
    }
}
