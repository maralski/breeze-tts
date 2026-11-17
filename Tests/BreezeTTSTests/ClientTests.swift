// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import BreezeTTS

@Test func checkpointNamesTheFilesAFolderLacks() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("breeze-checkpoint-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder.appendingPathComponent("audio_tokenizer"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let checkpoint = BreezeCheckpoint(directory: folder)
    #expect(checkpoint.missingFiles == BreezeCheckpoint.synthesisFiles)
    for file in ["tokenizer.json", "audio_tokenizer/model.safetensors", "model.safetensors.index.json"] {
        try Data().write(to: folder.appendingPathComponent(file))
    }
    #expect(checkpoint.missingFiles == ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    #expect(!checkpoint.isComplete)
    do {
        _ = try BreezeTTSModel(directory: folder)
        Issue.record("loading a folder without the weights did not throw")
    } catch let error as BreezeTTSError {
        #expect(error.description == "bad checkpoint: \(folder.path) lacks model-00001-of-00002.safetensors, model-00002-of-00002.safetensors")
    }
}

@Test func checkpointIsFoundByEnvironmentThenInTheHuggingFaceCache() throws {
    let cache = FileManager.default.temporaryDirectory.appendingPathComponent("breeze-cache-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: cache) }
    #expect(BreezeCheckpoint.find(environment: ["HF_HUB_CACHE": cache.path]) == nil)
    let snapshots = cache.appendingPathComponent("models--BreezeBlue--Breeze-TTS-2/snapshots")
    let partial = snapshots.appendingPathComponent("newer"), complete = snapshots.appendingPathComponent("older")
    for (folder, files, age) in [(complete, BreezeCheckpoint.synthesisFiles + BreezeCheckpoint.cloningFiles, -100.0), (partial, ["tokenizer.json"], 0.0)] {
        try FileManager.default.createDirectory(at: folder.appendingPathComponent("audio_tokenizer"), withIntermediateDirectories: true)
        for file in files { try Data().write(to: folder.appendingPathComponent(file)) }
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(age)], ofItemAtPath: folder.path)
    }
    #expect(BreezeCheckpoint.find(environment: ["HF_HUB_CACHE": cache.path])?.directory.lastPathComponent == "older")
    #expect(BreezeCheckpoint.find(environment: ["HF_HOME": "/nonexistent", "HF_HUB_CACHE": cache.path])?.isComplete == true)
    #expect(BreezeCheckpoint.find(environment: ["HF_HOME": cache.deletingLastPathComponent().path])?.directory.lastPathComponent != "older")
    try FileManager.default.removeItem(at: complete)
    #expect(BreezeCheckpoint.find(environment: ["HF_HUB_CACHE": cache.path])?.directory.lastPathComponent == "newer")
    #expect(BreezeCheckpoint.find(environment: ["BREEZE_WEIGHTS": "/somewhere/else", "HF_HUB_CACHE": cache.path])?.directory.path == "/somewhere/else")
}

@Test func optionsRejectSettingsTheSamplerCannotUse() {
    func rejects(_ change: (inout BreezeSynthesisOptions) -> Void) -> Bool {
        var o = BreezeSynthesisOptions()
        change(&o)
        return (try? o.validate()) == nil
    }
    #expect(!rejects { _ in })
    #expect(rejects { $0.maxFrames = 0 })
    #expect(rejects { $0.temperature = 0 })
    #expect(rejects { $0.temperature = .nan })
    #expect(rejects { $0.topK = 0 })
    #expect(rejects { $0.topK = 2049 })
    #expect(!rejects { $0.topK = 2048 })
    #expect(rejects { $0.repetitionPenalty = -1 })
    #expect(rejects { $0.fragmentLength = 0 })
}

@Test func optionsDefaultToTheReferenceSampling() {
    let o = BreezeSynthesisOptions()
    #expect(o.guidance == nil && o.maxFrames == 1500 && o.temperature == 0.9 && o.topK == 50 && o.repetitionPenalty == 1.1 && !o.greedy)
    #expect(BreezeTask.plain.recommendedGuidance == 1)
    #expect(BreezeTask.voiceDesign(instruction: "Calm.").recommendedGuidance == 4)
}

@Test func referenceVoiceReportsItsLength() throws {
    let voice = try BreezeReferenceVoice(transcript: "Hello.", codes: [[Int16]](repeating: [Int16](repeating: 0, count: 16), count: 125))
    #expect(voice.seconds == 10)
    #expect(voice.seconds < BreezeReferenceVoice.recommendedMaxSeconds)
}

/// Fragment 0's frames up to its trailing silence become the voice of the rest: a design
/// continues as a direction with its instruction, a plain request as a clone.
@Test func laterFragmentsCarryFragmentZerosVoice() throws {
    let frame = [Int16](repeating: 5, count: 16)
    let spoken = [Float](repeating: 0.2, count: 10 * 1920), silent = [Float](repeating: 0, count: 10 * 1920)
    let codes = [[Int16]](repeating: frame, count: 20)
    let design = try BreezeTTSModel.carriedTask(.voiceDesign(instruction: "Warm."), fragment: "First part.", codes: codes, pcm: spoken + silent)
    guard case .voiceDirection(let voice, let instruction) = design else { Issue.record("a design carried as \(String(describing: design))"); return }
    #expect(instruction == "Warm.")
    #expect(voice.transcript == "First part.")
    // 10 spoken frames, plus the 0.3 s tail trimTrailingSilence keeps, rounded up to whole frames.
    #expect(voice.frames == 14)
    let plain = try BreezeTTSModel.carriedTask(.plain, fragment: "First part.", codes: codes, pcm: spoken + silent)
    guard case .voiceClone = plain else { Issue.record("a plain request carried as \(String(describing: plain))"); return }
    #expect(try BreezeTTSModel.carriedTask(.plain, fragment: "Silence.", codes: [], pcm: []) == nil)
    let voice2 = try BreezeReferenceVoice(transcript: "Hi.", codes: [frame])
    #expect(try BreezeTTSModel.carriedTask(.voiceClone(voice2), fragment: "x", codes: codes, pcm: spoken) == nil)
}

@Test func trimSilenceKeepsSpeechAndAPad() {
    let rate = 24_000
    let speech = [Float](repeating: 0, count: rate) + (0..<rate).map { sin(Float($0) * 0.05) * 0.5 } + [Float](repeating: 0, count: rate)
    let trimmed = trimSilence(speech, sampleRate: rate)
    #expect(abs(trimmed.count - (rate + 2 * rate / 20)) <= rate / 100)
    #expect(trimSilence([Float](repeating: 0, count: rate), sampleRate: rate).isEmpty)
    #expect(trimTrailingSilence(speech, sampleRate: rate).count == 2 * rate + rate * 3 / 10)
}
