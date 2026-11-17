// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import XCTest
@testable import BreezeTTS

/// Stage-by-stage parity against the reference goldens. Runs when BREEZE_GOLDENS points at the
/// goldens directory (the checkpoint comes from BREEZE_WEIGHTS or the Hugging Face cache); skips
/// otherwise. Every fp32 tensor needs cosine >= 0.9999 on every row, and every token stream must
/// match the reference except at a draw the two logit sets cannot tell apart.
final class ParityTests: XCTestCase {
    nonisolated(unsafe) static var parity: BreezeParity?

    func check(_ name: String) throws {
        let env = ProcessInfo.processInfo.environment
        guard let goldens = env["BREEZE_GOLDENS"] else { throw XCTSkip("set BREEZE_GOLDENS to run the parity tests") }
        guard let weights = FrameBench.defaultWeights(environment: env) else { throw XCTSkip("no Breeze-TTS-2 checkpoint") }
        let parity = try Self.parity ?? BreezeParity(weights: weights)
        Self.parity = parity
        let dir = URL(fileURLWithPath: goldens).appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: dir.path) else { throw XCTSkip("no \(name) goldens") }
        let report = try parity.run(caseDirectory: dir)
        for line in report.tensors where !line.passed { XCTFail(line.description) }
        for line in report.streams where !line.passed { XCTFail(line.description) }
        XCTAssertGreaterThan(report.tensors.count, 6, "the case compared its tensors")
    }

    func testDesignEnglish() throws { try check("design_en") }
    func testPlainEnglish() throws { try check("plain_en") }
    func testDesignChinese() throws { try check("design_zh") }
    func testCloneEnglish() throws { try check("clone_en") }
    func testDirectionEnglish() throws { try check("direction_en") }
    /// One 337-token segment from tools/reference/make_long_text_golden.py: the sliding window.
    func testLongTextEncoder() throws { try check("long_text") }

    /// The comparison the whole report rests on must fail a zeroed row and a NaN.
    func testComparisonCatchesBrokenRows() {
        let want: [Float] = [1, 2, 3, 4, 5, 6]
        XCTAssertEqual(BreezeParity.compare(want, want, width: 3).cosine, 1, accuracy: 1e-12)
        XCTAssertLessThan(BreezeParity.compare([0, 0, 0, 4, 5, 6], want, width: 3).cosine, 0.9999)
        XCTAssertLessThan(BreezeParity.compare([1, 2, .nan, 4, 5, 6], want, width: 3).cosine, 0.9999)
    }
}
