// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import BreezeTTS

final class SamplingTests: GPUTestCase {
    enum Logits { case smooth, tied, peaked }

    /// One logit row. `tied` draws from a handful of values so top-k meets ties at its cut.
    func logitRow(_ rng: inout RNG, _ n: Int, _ kind: Logits) -> [Float] {
        switch kind {
        case .smooth: return rng.floats(n, scale: 6)
        case .tied: return (0..<n).map { _ in Float(Int(rng.bits() % 9)) * 0.75 - 3 }
        case .peaked: return (0..<n).map { _ in rng.next() * 3 + (rng.bits() % 40 == 0 ? 9 : 0) }
        }
    }

    /// A chain of draws on the GPU against `SamplingRule`, token for token. Host and GPU `exp`
    /// differ in the last bits, so a draw whose uniform lands within 1e-5 of a cumulative boundary
    /// may take either side; those are counted and must stay rare. The backbone chain shares one
    /// history set, which every draw extends.
    func chain(_ kind: SamplerKind, draws: Int, logits: Logits, guided: Bool, greedy: Bool, seed: UInt64) throws {
        let n = kind == .backbone ? BreezeShape.backboneVocab : BreezeShape.codebookVocab, ld = 2052
        var rng = RNG(state: seed)
        var settings = SamplerSettings()
        settings.greedy = greedy
        let guidance: Float? = guided ? 4 : nil
        var rows: [[Float]] = [], uniforms: [Float] = []
        for i in 0..<draws {
            let cond = logitRow(&rng, n, logits)
            rows.append(cond)
            // A guided tie needs equal (cond, uncond) pairs: copy some tokens' pairs onto others.
            var uncond = logitRow(&rng, n, logits)
            if guided, logits == .tied { for j in stride(from: 5, to: n, by: 3) { rows[rows.count - 1][j] = cond[j - 5]; uncond[j] = uncond[j - 5] } }
            if guided { rows.append(uncond) }
            uniforms.append(i == 0 ? 0 : i == 1 ? 0.99999994 : rng.unit())
        }
        let perDraw = guided ? 2 : 1
        var flat = [Float](repeating: .nan, count: rows.count * ld)
        for (r, row) in rows.enumerated() { flat.replaceSubrange((r * ld)..<(r * ld + n), with: row) }
        let lb = try f32Buffer(flat), ub = try f32Buffer(uniforms)
        let codes = try u32Buffer([UInt32](repeating: 0xFFFF_FFFF, count: draws))
        let seen = try u32Buffer([UInt32](repeating: 0, count: 2052)), status = try u32Buffer([0])
        try run { e in
            for i in 0..<draws {
                e.sample(kind, logits: ActView(lb, ld: ld).row(i * perDraw), guidance: guidance, settings: settings, uniforms: ub, uniformsOffset: 0,
                         codes: codes, codesOffset: 0, slot: i, seen: seen, status: status)
            }
        }
        let got = Array(UnsafeBufferPointer(start: codes.contents().bindMemory(to: UInt32.self, capacity: draws), count: draws)).map(Int.init)
        XCTAssertEqual(status.contents().load(as: UInt32.self), 0, "status")
        var history = Set<Int>(), boundary = 0
        let what = "\(kind) \(logits) guided \(guided) greedy \(greedy)"
        for i in 0..<draws {
            let d = SamplingRule.draw(kind, cond: rows[i * perDraw], uncond: guided ? rows[i * perDraw + 1] : nil, guidance: guidance ?? 1, seen: history, settings: settings, uniform: uniforms[i])
            if got[i] != d.token {
                let margin = 1e-5 * (d.cumulative.last ?? 0)
                let low = d.cumulative.firstIndex { $0 > d.target - margin }, high = d.cumulative.firstIndex { $0 > d.target + margin } ?? d.token
                guard !greedy, got[i] == low || got[i] == high else {
                    XCTFail("\(what): draw \(i) got \(got[i]) want \(d.token)")
                    return
                }
                boundary += 1
            }
            // The chain follows the GPU's token so one boundary draw cannot derail the rest.
            if kind == .backbone { history.insert(got[i]) }
        }
        XCTAssertLessThanOrEqual(boundary, max(1, draws / 100), "\(what): draws on a rounding boundary")
        if kind == .backbone {
            let flags = UnsafeBufferPointer(start: seen.contents().bindMemory(to: UInt32.self, capacity: 2052), count: 2052)
            XCTAssertEqual(Set(flags.indices.filter { flags[$0] != 0 }), history, "\(what): history set")
            XCTAssertGreaterThan(history.count, 1)
        }
        XCTAssertTrue(got.allSatisfy { $0 < 2048 || (kind == .backbone && $0 == BreezeShape.endOfSpeech) }, "\(what): a reserved id was drawn")
    }

    func testBackboneSampler() throws {
        var seed: UInt64 = 1
        for logits in [Logits.smooth, .tied, .peaked] {
            for guided in [false, true] {
                try chain(.backbone, draws: 400, logits: logits, guided: guided, greedy: false, seed: seed)
                seed += 1
            }
        }
    }

    func testDepthSampler() throws {
        var seed: UInt64 = 100
        for logits in [Logits.smooth, .tied, .peaked] {
            for guided in [false, true] {
                try chain(.depth, draws: 400, logits: logits, guided: guided, greedy: false, seed: seed)
                seed += 1
            }
        }
    }

    func testGreedy() throws {
        try chain(.backbone, draws: 200, logits: .tied, guided: true, greedy: true, seed: 7)
        try chain(.backbone, draws: 200, logits: .smooth, guided: false, greedy: true, seed: 8)
        try chain(.depth, draws: 200, logits: .tied, guided: false, greedy: true, seed: 9)
        try chain(.depth, draws: 200, logits: .smooth, guided: true, greedy: true, seed: 10)
    }

    /// The rule's discrete choices on hand-built rows, on the CPU and on the GPU.
    func testRuleEdges() throws {
        var s = SamplerSettings()
        s.topK = 2
        // Three equal maxima: the backbone keeps all three, the depth sampler ids 3 and 5.
        var row = [Float](repeating: -4, count: 2052)
        for i in [3, 5, 9] { row[i] = 2 }
        func gpu(_ kind: SamplerKind, _ row: [Float], _ u: Float, seenIDs: [Int] = [], settings: SamplerSettings) throws -> (Int, UInt32) {
            let lb = try f32Buffer(row), ub = try f32Buffer([u]), codes = try u32Buffer([0]), status = try u32Buffer([0])
            var flags = [UInt32](repeating: 0, count: 2052)
            for i in seenIDs { flags[i] = 1 }
            let seen = try u32Buffer(flags)
            try run { e in e.sample(kind, logits: ActView(lb, ld: 2052), guidance: nil, settings: settings, uniforms: ub, uniformsOffset: 0, codes: codes, codesOffset: 0, slot: 0, seen: seen, status: status) }
            return (Int(codes.contents().load(as: UInt32.self)), status.contents().load(as: UInt32.self))
        }
        for (u, backbone, depth) in [(Float(0.1), 3, 3), (0.4, 5, 3), (0.5, 5, 5), (0.9, 9, 5)] {
            XCTAssertEqual(SamplingRule.draw(.backbone, cond: row, settings: s, uniform: u).token, backbone)
            XCTAssertEqual(SamplingRule.draw(.depth, cond: Array(row[..<2051]), settings: s, uniform: u).token, depth)
            XCTAssertEqual(try gpu(.backbone, row, u, settings: s).0, backbone, "u \(u)")
            XCTAssertEqual(try gpu(.depth, row, u, settings: s).0, depth, "u \(u)")
        }
        // The penalty moves a seen maximum below the others; reserved ids never win.
        s.greedy = true
        row[2049] = 50
        XCTAssertEqual(try gpu(.backbone, row, 0, settings: s).0, 3)
        XCTAssertEqual(try gpu(.backbone, row, 0, seenIDs: [3], settings: s).0, 5)
        XCTAssertEqual(SamplingRule.draw(.backbone, cond: row, seen: [3, 5], settings: s, uniform: 0).token, 9)
        // Unusable logits raise the status bit and still leave an in-range code.
        row[100] = .nan
        let (token, status) = try gpu(.depth, row, 0.5, settings: SamplerSettings())
        XCTAssertEqual(status & 1, 1)
        XCTAssertLessThan(token, 2051)
    }

    /// One code's row and a frame's 16-row sum from the bf16 table, codes read on the GPU.
    func testEmbedCodes() throws {
        let width = 64, vocab = BreezeShape.codebookVocab, books = BreezeShape.codebooks, frames = 3
        var rng = RNG(state: 21)
        let (table, tf) = try bf16Tensor(rng.floats(books * vocab * width))
        let codes = (0..<(frames * books)).map { _ in UInt32(rng.bits() % 2051) }
        let cb = try u32Buffer(codes)
        let sum = try nanBuffer(floats: frames * width), same = try nanBuffer(floats: 2 * 2 * width), one = try nanBuffer(floats: 2 * width)
        try run { e in
            e.embedCodes(table: table, width: width, vocab: vocab, codes: cb, codesOffset: 0, firstCodebook: 0, count: books, rows: frames, rowStride: books, ActView(sum, ld: width))
            // Frame 1 into rows 1 and 3 of a 4-row buffer, and its codebook 7 alone into two rows.
            e.embedCodes(table: table, width: width, vocab: vocab, codes: cb, codesOffset: books * 4, firstCodebook: 0, count: books, rows: 2, ActView(same, ld: 2 * width).row(0).column(width))
            e.embedCodes(table: table, width: width, vocab: vocab, codes: cb, codesOffset: books * 4, firstCodebook: 7, count: 1, rows: 2, ActView(one, ld: width))
        }
        func rowOf(_ frame: Int, _ q: Int) -> ArraySlice<Float> {
            let at = (Int(codes[frame * books + q]) + q * vocab) * width
            return tf[at..<(at + width)]
        }
        func frameSum(_ frame: Int) -> [Float] {
            var acc = [Float](repeating: 0, count: width)
            for q in 0..<books { for (d, v) in rowOf(frame, q).enumerated() { acc[d] += v } }
            return acc
        }
        XCTAssertEqual(sum.floats(count: frames * width), (0..<frames).flatMap(frameSum), "sum over frames")
        let s = same.floats(count: 4 * width)
        XCTAssertEqual(Array(s[width..<(2 * width)]), frameSum(1))
        XCTAssertEqual(Array(s[(3 * width)...]), frameSum(1))
        XCTAssertTrue(s[..<width].allSatisfy(\.isNaN), "rows between the strided outputs stay untouched")
        XCTAssertEqual(one.floats(count: 2 * width), Array(rowOf(1, 7)) + Array(rowOf(1, 7)), "single row")
    }
}
