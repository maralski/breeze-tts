// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import XCTest
@testable import BreezeTTS

/// The public model end to end on the real checkpoint (BREEZE_WEIGHTS or the Hugging Face cache);
/// the PCM comparisons also need BREEZE_GOLDENS. Skips without them.
final class ModelTests: XCTestCase {
    nonisolated(unsafe) static var shared: BreezeTTSModel?

    func model() throws -> BreezeTTSModel {
        guard let dir = TestEnvironment.weightsDirectory else { throw XCTSkip("set BREEZE_WEIGHTS to the Breeze-TTS-2 snapshot") }
        if let m = Self.shared { return m }
        let m = try BreezeTTSModel(directory: dir)
        Self.shared = m
        return m
    }

    static let text = "Thank you for calling. How can I help you today?"

    func greedy() -> BreezeSynthesisOptions {
        var o = BreezeSynthesisOptions()
        o.greedy = true
        return o
    }

    func testReportCountsWhatWasDelivered() throws {
        let m = try model()
        var chunks: [[Float]] = [], fragments = Set<Int>()
        let t0 = Date()
        let reports = try m.synthesize(try BreezeRequest(text: Self.text), options: greedy()) { fragment, chunk in
            fragments.insert(fragment)
            chunks.append(chunk)
        }
        let wall = Date().timeIntervalSince(t0)
        XCTAssertEqual(reports.count, 1)
        let r = reports[0]
        XCTAssertTrue(r.stopped)
        XCTAssertEqual(r.frames, chunks.count)
        XCTAssertEqual(r.frameSeconds.count, r.frames)
        XCTAssertTrue(chunks.allSatisfy { $0.count == 1920 })
        XCTAssertEqual(fragments, [0])
        let first = try XCTUnwrap(r.firstAudioSeconds)
        XCTAssertGreaterThan(first, r.prefillSeconds)
        XCTAssertLessThan(first, wall)
        XCTAssertGreaterThan(chunks.joined().map(abs).max() ?? 0, 0.05, "the audio is not silent")
    }

    /// A throw from the callback stops generation, comes back out, and leaves the model ready:
    /// the next request gives what a fresh one would.
    func testThrowingFromTheCallbackCancels() throws {
        struct Stop: Error {}
        let m = try model()
        let request = try BreezeRequest(text: Self.text)
        var delivered = 0
        XCTAssertThrowsError(try m.synthesize(request, options: greedy()) { _, _ in
            delivered += 1
            if delivered == 3 { throw Stop() }
        }) { XCTAssertTrue($0 is Stop) }
        XCTAssertEqual(delivered, 3)
        var after: [Float] = [], again: [Float] = []
        _ = try m.synthesize(request, options: greedy()) { after += $1 }
        _ = try m.synthesize(request, options: greedy()) { again += $1 }
        XCTAssertGreaterThan(after.count, 3 * 1920)
        XCTAssertEqual(after, again)
    }

    /// Breaking out of the stream cancels the generation behind it; the model then serves the
    /// next request long before an uncancelled 120 s generation could have finished.
    func testStreamCancellationStopsGeneration() async throws {
        let m = try model()
        var o = BreezeSynthesisOptions()
        o.maxFrames = 1500
        let long = try BreezeRequest(text: String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 8))
        var received = 0
        for try await _ in m.stream(long, options: o) {
            received += 1
            if received == 2 { break }
        }
        XCTAssertEqual(received, 2)
        let t0 = Date()
        let reports = try m.synthesize(try BreezeRequest(text: "Hello."), options: greedy()) { _, _ in }
        XCTAssertEqual(reports.count, 1)
        XCTAssertLessThan(Date().timeIntervalSince(t0), 10, "the cancelled stream still held the model")
    }

    func testGuidanceThePlainTaskRejectsThrowsBeforeAudio() throws {
        let m = try model()
        var o = BreezeSynthesisOptions()
        o.guidance = 4
        var chunks = 0
        XCTAssertThrowsError(try m.synthesize(try BreezeRequest(text: Self.text), options: o) { _, _ in chunks += 1 })
        XCTAssertEqual(chunks, 0)
    }

    /// Greedy draws from text to the WAV the CLI writes, against the reference's codec output for
    /// its own greedy codes (codec/pcm_full.f32 of make_goldens.py).
    func testGreedyAudioEqualsTheReference() throws {
        guard let goldens = ProcessInfo.processInfo.environment["BREEZE_GOLDENS"].map(URL.init(fileURLWithPath:)) else {
            throw XCTSkip("set BREEZE_GOLDENS to compare against the reference PCM")
        }
        let m = try model()
        for name in ["plain_en", "design_en", "design_zh", "clone_en", "direction_en"] {
            let golden = try GoldenCase(directory: goldens.appendingPathComponent(name))
            var pcm: [Float] = []
            let reports = try m.synthesize(try golden.request(), options: greedy()) { pcm += $1 }
            let want = try CodecGoldens.floats(goldens.appendingPathComponent("\(name)/codec/pcm_full.f32"))
            XCTAssertTrue(reports[0].stopped, name)
            XCTAssertEqual(pcm.count, want.count, name)
            guard pcm.count == want.count else { continue }
            let worst = zip(pcm, want).map { abs($0 - $1) }.max()!
            print("\(name): \(pcm.count) samples, max abs difference \(worst)")
            XCTAssertLessThanOrEqual(worst, 2e-5, name)
        }
    }

    /// The same comparison through the breeze-tts binary and its WAV file, when the build put
    /// the binary beside the test bundle.
    func testCLIWavEqualsTheReference() throws {
        guard let goldens = ProcessInfo.processInfo.environment["BREEZE_GOLDENS"] else { throw XCTSkip("set BREEZE_GOLDENS") }
        guard let weights = TestEnvironment.weightsDirectory else { throw XCTSkip("set BREEZE_WEIGHTS") }
        let binary = Bundle(for: Self.self).bundleURL.deletingLastPathComponent().appendingPathComponent("breeze-tts")
        guard FileManager.default.isExecutableFile(atPath: binary.path) else { throw XCTSkip("no breeze-tts beside the test bundle; run swift build first") }
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("breeze-cli-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: out) }
        let process = Process()
        process.executableURL = binary
        process.arguments = ["--say", Self.text, "--greedy", "--weights", weights.path, "--out", out.path]
        process.standardOutput = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        let data = try Data(contentsOf: out)
        XCTAssertEqual(String(decoding: data[0..<4], as: UTF8.self), "RIFF")
        let pcm = data[44...].withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let want = try CodecGoldens.floats(URL(fileURLWithPath: goldens).appendingPathComponent("plain_en/codec/pcm_full.f32"))
        XCTAssertEqual(pcm.count, want.count)
        guard pcm.count == want.count else { return }
        let worst = zip(pcm, want).map { abs($0 - $1) }.max()!
        print("breeze-tts --say plain_en: max abs difference \(worst)")
        XCTAssertLessThanOrEqual(worst, 2e-5)
    }
}
