// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import BreezeTTS

final class AttentionTests: GPUTestCase {
    /// A cache layer for `sequences` sequences, every float NaN until a kernel writes it.
    func nanCache(sequences: Int, kvHeads: Int, capacity: Int, headDim: Int) throws -> KVCacheLayer {
        let stride = kvHeads * capacity * headDim
        let k = try nanBuffer(floats: sequences * stride), v = try nanBuffer(floats: sequences * stride)
        return KVCacheLayer(k: GPUTensor(buffer: k, offset: 0, count: sequences * stride), v: GPUTensor(buffer: v, offset: 0, count: sequences * stride),
                            capacity: capacity, seqStride: stride)
    }

    static func rmsScale(_ v: ArraySlice<Float>, eps: Float) -> Float {
        1 / (dot(v, v) / Float(v.count) + eps).squareRoot()
    }

    /// norm_rope_kv over two sequences at unequal positions: q in place, k and v in the cache at
    /// each row's position, nothing else touched.
    func normRopeCase(headDim D: Int, hq: Int, hkv: Int, rows M: Int, positions: [Int], mode: Int) throws {
        let B = positions.count, ld = (hq + 2 * hkv) * D + 8, cap = positions.max()! + M + 3, eps: Float = 1e-6
        var rng = RNG(state: UInt64(D + hq * 7 + M * 3 + mode))
        let qkv = rng.floats(B * M * ld, scale: 2)
        let invFreq = RopeFrequencies.plain(theta: 1e6, dim: D).values
        let (w, wf) = try bf16Tensor(rng.floats(2 * D))
        let cache = try nanCache(sequences: B, kvHeads: hkv, capacity: cap, headDim: D)
        let qb = try f32Buffer(qkv), fb = try f32Buffer(invFreq)
        let norm: QKNorm = mode == 0 ? .none : mode == 1 ? .weight(w, eps: eps) : .onePlusWeight(w, eps: eps)
        try run { e in
            e.normRopeKV(ActView(qb, ld: ld), block: RowBlock(rows: M, positions: positions), heads: hq, kvHeads: hkv, headDim: D, cache: cache, invFreq: fb, norm: norm)
        }
        var wantQ = qkv
        var wantK = [Float](repeating: .nan, count: B * hkv * cap * D), wantV = wantK
        for s in 0..<B {
            for m in 0..<M {
                let row = (s * M + m) * ld, pos = positions[s] + m
                for slot in 0..<(hq + hkv) {
                    var head = Array(qkv[(row + slot * D)..<(row + (slot + 1) * D)])
                    if mode != 0 {
                        let scale = Self.rmsScale(head[...], eps: eps), w0 = slot < hq ? 0 : D
                        for d in 0..<D { head[d] = head[d] * scale * ((mode == 2 ? 1 : 0) + wf[w0 + d]) }
                    }
                    head = Self.rotateHalf(head, pos: pos, invFreq: invFreq)
                    if slot < hq {
                        wantQ.replaceSubrange((row + slot * D)..<(row + (slot + 1) * D), with: head)
                    } else {
                        let at = ((s * hkv + slot - hq) * cap + pos) * D
                        wantK.replaceSubrange(at..<(at + D), with: head)
                        let v = row + (slot + hkv) * D
                        wantV.replaceSubrange(at..<(at + D), with: qkv[v..<(v + D)])
                    }
                }
            }
        }
        func same(_ got: [Float], _ want: [Float], _ what: String) {
            XCTAssertEqual(got.map(\.isNaN), want.map(\.isNaN), "\(what): written positions, D \(D) mode \(mode)")
            assertClose(got.map { $0.isNaN ? 0 : $0 }, want.map { $0.isNaN ? 0 : $0 }, atol: 2e-5, "\(what) D \(D) mode \(mode)")
        }
        same(qb.floats(count: qkv.count), wantQ, "q in place")
        same(cache.k.buffer.floats(count: wantK.count), wantK, "k cache")
        same(cache.v.buffer.floats(count: wantV.count), wantV, "v cache")
    }

    func testNormRopeKV() throws {
        for mode in 0..<3 {
            try normRopeCase(headDim: 128, hq: 16, hkv: 8, rows: 3, positions: [300, 41], mode: mode)
            try normRopeCase(headDim: 256, hq: 4, hkv: 1, rows: 5, positions: [0], mode: mode)
        }
        try normRopeCase(headDim: 128, hq: 16, hkv: 8, rows: 1, positions: [2040, 1999], mode: 1)
    }

    /// (a): prefill of two sequences of different lengths, each alone, then decode steps of both
    /// together, against a CPU reference over the whole of each sequence. The lengths pass the
    /// 512 keys MingTTS's kernel stopped at.
    func causalCase(lengths: [Int], decode: Int, split: Int) throws {
        let hq = 4, hkv = 2, D = 128, ld = (hq + 2 * hkv) * D, B = lengths.count
        let shipped = Encoder.causalDecodeSplit
        Encoder.causalDecodeSplit = split
        defer { Encoder.causalDecodeSplit = shipped }
        var rng = RNG(state: UInt64(lengths[0] * 31 + decode + split))
        let invFreq = RopeFrequencies.plain(theta: 1e6, dim: D).values
        let fb = try f32Buffer(invFreq)
        let cap = lengths.max()! + decode + 5
        let cache = try nanCache(sequences: B, kvHeads: hkv, capacity: cap, headDim: D)
        let total = lengths.map { $0 + decode }
        let seqs = total.map { rng.floats($0 * ld) }
        var got = total.map { [Float](repeating: 0, count: $0 * hq * D) }
        // Prefill, one sequence at a time.
        for s in 0..<B {
            let qb = try f32Buffer(Array(seqs[s][..<(lengths[s] * ld)]))
            let out = try nanBuffer(floats: lengths[s] * hq * D)
            let block = RowBlock(rows: lengths[s], positions: [0])
            try run { e in
                e.normRopeKV(ActView(qb, ld: ld), block: block, heads: hq, kvHeads: hkv, headDim: D, cache: cache.sequences(from: s), invFreq: fb, norm: .none)
                e.attentionCausal(ActView(qb, ld: ld), block: block, heads: hq, kvHeads: hkv, cache: cache.sequences(from: s), ActView(out, ld: hq * D))
            }
            got[s].replaceSubrange(0..<(lengths[s] * hq * D), with: out.floats(count: lengths[s] * hq * D))
        }
        for step in 0..<decode {
            let rows = (0..<B).flatMap { s in Array(seqs[s][((lengths[s] + step) * ld)..<((lengths[s] + step + 1) * ld)]) }
            let qb = try f32Buffer(rows + [Float](repeating: .nan, count: ld))
            let out = try nanBuffer(floats: B * hq * D)
            let block = RowBlock(rows: 1, positions: lengths.map { $0 + step })
            try run(concurrent: true) { e in
                e.normRopeKV(ActView(qb, ld: ld), block: block, heads: hq, kvHeads: hkv, headDim: D, cache: cache, invFreq: fb, norm: .none)
                e.attentionCausal(ActView(qb, ld: ld), block: block, heads: hq, kvHeads: hkv, cache: cache, ActView(out, ld: hq * D))
            }
            let o = out.floats(count: B * hq * D)
            for s in 0..<B { got[s].replaceSubrange(((lengths[s] + step) * hq * D)..<((lengths[s] + step + 1) * hq * D), with: o[(s * hq * D)..<((s + 1) * hq * D)]) }
        }
        for s in 0..<B {
            let x = seqs[s]
            func head(_ pos: Int, _ slot: Int, rotated: Bool) -> [Float] {
                let v = Array(x[(pos * ld + slot * D)..<(pos * ld + (slot + 1) * D)])
                return rotated ? Self.rotateHalf(v, pos: pos, invFreq: invFreq) : v
            }
            let keys = (0..<hkv).map { h in (0..<total[s]).map { head($0, hq + h, rotated: true) } }
            let values = (0..<hkv).map { h in (0..<total[s]).map { head($0, hq + hkv + h, rotated: false) } }
            var want = [Float](repeating: 0, count: total[s] * hq * D)
            for pos in 0..<total[s] {
                for h in 0..<hq {
                    let q = head(pos, h, rotated: true), kvh = h / (hq / hkv)
                    let scores = (0...pos).map { Self.dot(q[...], keys[kvh][$0][...]) / Float(D).squareRoot() }
                    want.replaceSubrange(((pos * hq + h) * D)..<((pos * hq + h + 1) * D), with: Self.attend(scores, Array(values[kvh][...pos])))
                }
            }
            assertClose(got[s], want, atol: 2e-5, "causal attention sequence \(s) of lengths \(lengths), split \(split)")
        }
    }

    func testCausalAttention() throws {
        try causalCase(lengths: [37, 9], decode: 2, split: 4)
        try causalCase(lengths: [700, 530], decode: 3, split: 4)
        try causalCase(lengths: [530], decode: 2, split: 8)
        try causalCase(lengths: [3, 1], decode: 12, split: 8)
        try causalCase(lengths: [200], decode: 1, split: 1)
    }

    /// (b): a frame's depth sequence, two rows then single rows, for `sequences` sequences in one
    /// dispatch, against the reference; the cache must end up holding the rotated keys and values.
    func shortCase(sequences B: Int, steps: Int) throws {
        let hq = 8, hkv = 2, D = 128, ld = (hq + 2 * hkv) * D, cap = 17, G = hq / hkv
        var rng = RNG(state: UInt64(B * 17 + steps))
        let rope = DecoderConfig.depth.rope.values
        let fb = try f32Buffer(rope)
        let cache = try nanCache(sequences: B, kvHeads: hkv, capacity: cap, headDim: D)
        let total = 2 + steps
        let seqs = (0..<B).map { _ in rng.floats(total * ld) }
        var got = [[Float]](repeating: [Float](repeating: 0, count: total * hq * D), count: B)
        var pos0 = 0
        for rows in [2] + [Int](repeating: 1, count: steps) {
            let x = (0..<B).flatMap { s in Array(seqs[s][(pos0 * ld)..<((pos0 + rows) * ld)]) }
            let qb = try f32Buffer(x), out = try nanBuffer(floats: B * rows * hq * D)
            try run { e in
                e.attentionShort(ActView(qb, ld: ld), sequences: B, rows: rows, pos0: pos0, heads: hq, kvHeads: hkv, cache: cache, invFreq: fb, ActView(out, ld: hq * D))
            }
            XCTAssertEqual(qb.floats(count: x.count), x, "attention_short leaves its input alone")
            let o = out.floats(count: B * rows * hq * D)
            for s in 0..<B {
                got[s].replaceSubrange((pos0 * hq * D)..<((pos0 + rows) * hq * D), with: o[(s * rows * hq * D)..<((s + 1) * rows * hq * D)])
            }
            pos0 += rows
        }
        for s in 0..<B {
            let x = seqs[s]
            func head(_ pos: Int, _ slot: Int, rotated: Bool) -> [Float] {
                let v = Array(x[(pos * ld + slot * D)..<(pos * ld + (slot + 1) * D)])
                return rotated ? Self.rotateHalf(v, pos: pos, invFreq: rope) : v
            }
            var want = [Float](repeating: 0, count: total * hq * D)
            for pos in 0..<total {
                for h in 0..<hq {
                    let q = head(pos, h, rotated: true), kvh = h / G
                    let scores = (0...pos).map { Self.dot(q[...], head($0, hq + kvh, rotated: true)[...]) / Float(D).squareRoot() }
                    want.replaceSubrange(((pos * hq + h) * D)..<((pos * hq + h + 1) * D), with: Self.attend(scores, (0...pos).map { head($0, hq + hkv + kvh, rotated: false) }))
                }
            }
            assertClose(got[s], want, atol: 2e-5, "short attention sequence \(s) of \(B)")
            for kvh in 0..<hkv {
                let at = (s * hkv + kvh) * cap * D * 4
                assertClose(cache.k.buffer.floats(count: total * D, offset: at), (0..<total).flatMap { head($0, hq + kvh, rotated: true) }, atol: 2e-5, "short attention k cache")
                XCTAssertEqual(cache.v.buffer.floats(count: total * D, offset: at), (0..<total).flatMap { head($0, hq + hkv + kvh, rotated: false) }, "short attention v cache")
            }
        }
    }

    func testShortAttention() throws {
        try shortCase(sequences: 1, steps: 14)
        try shortCase(sequences: 2, steps: 14)
    }

    /// (c): bidirectional attention, full and under Gemma's asymmetric window.
    func windowCase(length L: Int, window: (before: Int, after: Int)?) throws {
        let hq = 4, D = 256
        var rng = RNG(state: UInt64(L + (window?.before ?? 0)))
        let q = rng.floats(L * hq * D), keys = rng.floats(L * D), values = rng.floats(L * D)
        // Rows past the sequence are NaN: the window must stop at its end.
        let kb = try f32Buffer(keys + [Float](repeating: .nan, count: 8 * D)), vb = try f32Buffer(values + [Float](repeating: .nan, count: 8 * D))
        let qb = try f32Buffer(q), out = try nanBuffer(floats: L * hq * D)
        try run { e in
            e.attentionWindow(ActView(qb, ld: hq * D), rows: L, total: L, window: window, keys: GPUTensor(buffer: kb, offset: 0, count: L * D),
                              values: GPUTensor(buffer: vb, offset: 0, count: L * D), scale: 0.0625, ActView(out, ld: hq * D))
        }
        var want = [Float](repeating: 0, count: L * hq * D)
        for i in 0..<L {
            let lo = max(0, i - (window?.before ?? L)), hi = min(L - 1, i + (window?.after ?? L))
            for h in 0..<hq {
                let qh = q[((i * hq + h) * D)..<((i * hq + h + 1) * D)]
                let scores = (lo...hi).map { Self.dot(qh, keys[($0 * D)..<(($0 + 1) * D)]) * 0.0625 }
                want.replaceSubrange(((i * hq + h) * D)..<((i * hq + h + 1) * D), with: Self.attend(scores, (lo...hi).map { Array(values[($0 * D)..<(($0 + 1) * D)]) }))
            }
        }
        assertClose(out.floats(count: L * hq * D), want, atol: 2e-5, "window attention length \(L) window \(String(describing: window))")
    }

    func testWindowAttention() throws {
        try windowCase(length: 70, window: nil)
        try windowCase(length: 70, window: (255, 256))
        try windowCase(length: 600, window: (255, 256))
        try windowCase(length: 13, window: (2, 3))
    }
}
