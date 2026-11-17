// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import BreezeTTS

private let frame = [Int16](repeating: 7, count: 16)

@Test func requestKeepsTextAndStripsInstruction() throws {
    let request = try BreezeRequest(text: "  Hello. ", task: .voiceDesign(instruction: "\n Warm and calm.  "))
    #expect(request.text == "  Hello. ")
    #expect(request.task == .voiceDesign(instruction: "Warm and calm."))
}

@Test func requestRejectsEmptyFields() throws {
    let voice = try BreezeReferenceVoice(transcript: "Hi.", codes: [frame])
    #expect(throws: BreezeTTSError.self) { try BreezeRequest(text: "") }
    #expect(throws: BreezeTTSError.self) { try BreezeRequest(text: " \n\t") }
    #expect(throws: BreezeTTSError.self) { try BreezeRequest(text: "Hi.", task: .voiceDesign(instruction: "  ")) }
    #expect(throws: BreezeTTSError.self) { try BreezeRequest(text: "Hi.", task: .voiceDirection(voice, instruction: "")) }
}

@Test func requestRejectsAudioMarkersInText() throws {
    #expect(throws: BreezeTTSError.self) { try BreezeRequest(text: "a<|AUDIO|>b") }
    #expect(throws: BreezeTTSError.self) { try BreezeRequest(text: "a<|AUDIO|>\u{301}b") }
    #expect(throws: BreezeTTSError.self) { try BreezeRequest(text: "Hi.", task: .voiceDesign(instruction: "<|audio_eos|>")) }
    #expect(throws: BreezeTTSError.self) { try BreezeReferenceVoice(transcript: "x<|audio_eos|>", codes: [frame]) }
    #expect(try BreezeRequest(text: "[S1] <|AUDIO (laugh) <ins_bos>").text == "[S1] <|AUDIO (laugh) <ins_bos>")
}

@Test func requestReferenceVoiceValidatesCodes() throws {
    #expect(throws: BreezeTTSError.self) { try BreezeReferenceVoice(transcript: " ", codes: [frame]) }
    #expect(throws: BreezeTTSError.self) { try BreezeReferenceVoice(transcript: "Hi.", codes: []) }
    #expect(throws: BreezeTTSError.self) { try BreezeReferenceVoice(transcript: "Hi.", codes: [Array(frame.dropLast())]) }
    #expect(throws: BreezeTTSError.self) { try BreezeReferenceVoice(transcript: "Hi.", codes: [frame, frame.map { _ in 2048 }]) }
    #expect(throws: BreezeTTSError.self) { try BreezeReferenceVoice(transcript: "Hi.", codes: [frame.map { _ in -1 }]) }
    let voice = try BreezeReferenceVoice(transcript: " Hi there.\n", codes: [[Int16]](repeating: frame, count: 25))
    #expect(voice.transcript == "Hi there.")
    #expect(voice.seconds == 2.0)
}

@Test func requestReferenceVoiceRoundTripsAndValidatesOnDecode() throws {
    let voice = try BreezeReferenceVoice(transcript: "Hi.", codes: [frame, frame.map { $0 + 1 }])
    let data = try JSONEncoder().encode(voice)
    #expect(try JSONDecoder().decode(BreezeReferenceVoice.self, from: data) == voice)
    let bad = Data(#"{"transcript": "Hi.", "codes": [[1, 2]]}"#.utf8)
    #expect(throws: BreezeTTSError.self) { try JSONDecoder().decode(BreezeReferenceVoice.self, from: bad) }
}

@Test func requestTaskGuidance() throws {
    let voice = try BreezeReferenceVoice(transcript: "Hi.", codes: [frame])
    let tasks: [BreezeTask] = [.plain, .voiceDesign(instruction: "x"), .voiceClone(voice), .voiceDirection(voice, instruction: "x")]
    #expect(tasks.map(\.acceptsGuidance) == [false, true, false, true])
    #expect(tasks.map(\.recommendedGuidance) == [1, 4, 1, 4])
    #expect(tasks.map { $0.reference != nil } == [false, false, true, true])
}

@Test func requestVocalEventsAreTheDocumentedFour() {
    #expect(BreezeVocalEvent.allCases.map(\.english) == ["(laugh)", "(cough)", "(clears throat)", "(sigh)"])
    #expect(BreezeVocalEvent.allCases.map(\.chinese) == ["[笑]", "[咳嗽]", "[清嗓子]", "[叹气]"])
}
