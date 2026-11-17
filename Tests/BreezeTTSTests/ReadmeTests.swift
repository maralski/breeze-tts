// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import BreezeTTS
import Foundation
import XCTest

/// The README's library examples, as written there except for the paths, so they keep compiling
/// against the public API. They run when a checkpoint is found.
final class ReadmeTests: XCTestCase {
    func testLibraryExamplesRun() async throws {
        guard let checkpoint = BreezeCheckpoint.find(), checkpoint.isComplete else { throw XCTSkip("no complete Breeze-TTS-2 checkpoint") }
        let clipURL = try XCTUnwrap(ProcessInfo.processInfo.environment["BREEZE_GOLDENS"].map {
            URL(fileURLWithPath: $0).appendingPathComponent("ref_clip_en.wav")
        }, "set BREEZE_GOLDENS for the reference clip")

        let model = try BreezeTTSModel(directory: checkpoint.directory)
        let request = try BreezeRequest(
            text: "Welcome aboard. Your journey begins now.",
            task: .voiceDesign(instruction: "A warm, thoughtful young woman with a clear voice."))

        var streamed = 0
        for try await chunk in model.stream(request) {
            // chunk is [Float] PCM at BreezeTTSModel.sampleRate (24,000 Hz), mono: one 80 ms frame.
            streamed += chunk.count
        }

        var options = SynthesisOptions()
        options.seed = 7
        let reports = try model.synthesize(request, options: options) { fragment, chunk in
            try Task.checkCancellation()
            // play or store chunk
        }
        let finished = reports.allSatisfy(\.stopped)

        let (samples, rate) = try BreezeTTSModel.readMono(clipURL)
        let voice = try model.encodeReference(samples: trimSilence(samples, sampleRate: rate), sampleRate: rate,
                                              transcript: "Welcome aboard. Your journey begins now.")
        let cloned = try BreezeRequest(text: "Hello.", task: .voiceClone(voice))
        let directed = try BreezeRequest(text: "Hello.", task: .voiceDirection(voice, instruction: "Whisper it."))

        XCTAssertGreaterThan(streamed, 24_000)
        XCTAssertTrue(finished)
        XCTAssertGreaterThan(voice.frames, 20)
        for r in [cloned, directed] {
            XCTAssertTrue(try model.synthesize(r) { _, _ in }.allSatisfy(\.stopped))
        }
    }
}
