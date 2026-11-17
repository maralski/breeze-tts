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

    func harness() throws -> (BreezeParity, URL) {
        let env = ProcessInfo.processInfo.environment
        guard let goldens = env["BREEZE_GOLDENS"] else { throw XCTSkip("set BREEZE_GOLDENS to run the parity tests") }
        guard let weights = BreezeCheckpoint.find(environment: env)?.directory else { throw XCTSkip("no Breeze-TTS-2 checkpoint") }
        let parity = try Self.parity ?? BreezeParity(weights: weights)
        Self.parity = parity
        return (parity, URL(fileURLWithPath: goldens))
    }

    func check(_ name: String) throws {
        let (parity, goldens) = try harness()
        let dir = goldens.appendingPathComponent(name)
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

    /// The golden prompts are all shorter than `Kernels.gemmMinRows`; this runs one through the
    /// tiled GEMM that longer prompts take.
    func testDirectionThroughTiledGemm() throws {
        let saved = Kernels.gemmMinRows
        Kernels.gemmMinRows = 16
        defer { Kernels.gemmMinRows = saved }
        try check("direction_en")
    }

    /// The seeded uniforms give the same codes run after run, to end of speech.
    func testSeededRunRepeats() throws {
        let (parity, goldens) = try harness()
        let prompt = try GoldenCase(directory: goldens.appendingPathComponent("design_en")).referencePrompt()
        var options = SynthesisOptions()
        options.uniforms = .seeded(7)
        options.maxFrames = 300
        var runs: [[[Int]]] = []
        for _ in 0..<2 {
            var codes: [[Int]] = []
            let report = try parity.synth.run(prompt, options: options) { codes.append($0) }
            XCTAssertTrue(report.stoppedOnEndOfSpeech)
            XCTAssertEqual(report.frames, codes.count)
            XCTAssertEqual(report.frameSeconds.count, codes.count)
            runs.append(codes)
        }
        XCTAssertGreaterThan(runs[0].count, 10)
        XCTAssertEqual(runs[0], runs[1])
    }

    /// A prompt of 2040 positions leaves the backbone room for 7 frames (the reference stops
    /// before position 2047).
    func testPositionLimitStopsGeneration() throws {
        let (parity, _) = try harness()
        var rng = SplitMix64(seed: 3)
        let ids = [Int32(2)] + (1..<2040).map { _ in Int32(3 + rng.next() % 200_000) }
        var options = SynthesisOptions()
        options.greedy = true
        var frames = 0
        let report = try parity.synth.run(BreezePrompt(positive: .init(segments: [.text(ids)]), guidance: nil), options: options) { _ in frames += 1 }
        XCTAssertLessThanOrEqual(frames, 7)
        XCTAssertTrue(report.stoppedOnEndOfSpeech || frames == 7, "\(frames) frames, stopped \(report.stoppedOnEndOfSpeech)")
    }

    /// The comparison the whole report rests on must fail a zeroed row and a NaN.
    func testComparisonCatchesBrokenRows() {
        let want: [Float] = [1, 2, 3, 4, 5, 6]
        XCTAssertEqual(BreezeParity.compare(want, want, width: 3).cosine, 1, accuracy: 1e-12)
        XCTAssertLessThan(BreezeParity.compare([0, 0, 0, 4, 5, 6], want, width: 3).cosine, 0.9999)
        XCTAssertLessThan(BreezeParity.compare([1, 2, .nan, 4, 5, 6], want, width: 3).cosine, 0.9999)
    }
}
