// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import BreezeTTS

/// Written by Fixtures/text/make_text_fixtures.py from the reference's templates.prepare_inputs.
private struct PromptFixtures: Decodable {
    struct Branch: Decodable { var ids: [Int32]; var text_mask: [Bool]; var text_lengths: [Int] }
    struct Case: Decodable {
        var task: String
        var text: String
        var instruction: String?
        var transcript: String?
        var codes: [[Int16]]?
        var guidance: Float
        var error: String?
        var positive: Branch?
        var negative: Branch?
    }
    var cases: [Case]
}

private func request(_ c: PromptFixtures.Case) throws -> BreezeRequest {
    let voice = try c.transcript.map { try BreezeReferenceVoice(transcript: $0, codes: c.codes!) }
    let task: BreezeTask = switch c.task {
    case "plain": .plain
    case "voiceDesign": .voiceDesign(instruction: c.instruction!)
    case "voiceClone": .voiceClone(voice!)
    default: .voiceDirection(voice!, instruction: c.instruction!)
    }
    return try BreezeRequest(text: c.text, task: task)
}

private func check(_ branch: BreezePrompt.Branch, _ want: PromptFixtures.Branch, _ voice: BreezeReferenceVoice?, _ what: String) {
    #expect(branch.ids == want.ids, "\(what) ids")
    #expect(branch.rows.map { $0 == .text } == want.text_mask, "\(what) text mask")
    #expect(branch.count == want.ids.count, "\(what) count")
    let texts = branch.segments.compactMap { if case .text(let ids) = $0 { ids } else { nil } }
    #expect(texts.map(\.count) == want.text_lengths, "\(what) text lengths")
    #expect(texts.allSatisfy { $0.first == 2 }, "\(what) segments start with <bos>")
    let audio = branch.segments.compactMap { if case .audio(let codes) = $0 { codes } else { nil } }
    #expect(audio == (voice.map { [$0.codes] } ?? []), "\(what) audio segments")
}

@Test(.enabled(if: textTokenizer != nil, "needs the Breeze-TTS-2 tokenizer.json")) func textPromptMatchesReferenceTemplates() throws {
    let tokenizer = try #require(textTokenizer)
    let cases = try JSONDecoder().decode(PromptFixtures.self, from: Data(contentsOf: textFixture("prompt_fixtures.json"))).cases
    #expect(Set(cases.map(\.task)) == ["plain", "voiceDesign", "voiceClone", "voiceDirection"])
    #expect(cases.contains { $0.negative != nil && $0.task == "voiceDesign" } && cases.contains { $0.negative != nil && $0.task == "voiceDirection" })
    for c in cases {
        let what = "\(c.task) \(c.text.prefix(30).debugDescription) guidance \(c.guidance)"
        let request = try request(c)
        guard c.error == nil else {
            #expect(throws: BreezeTTSError.self, "\(what)") {
                try BreezePrompt(fragment: request.text, of: request, guidance: c.guidance, tokenizer: tokenizer)
            }
            continue
        }
        let prompt = try BreezePrompt(fragment: request.text, of: request, guidance: c.guidance, tokenizer: tokenizer)
        check(prompt.positive, try #require(c.positive), request.task.reference, what)
        #expect((prompt.guidance == nil) == (c.negative == nil), "\(what) guidance")
        if let guidance = prompt.guidance, let negative = c.negative {
            #expect(guidance.scale == c.guidance)
            check(guidance.negative, negative, request.task.reference, what + " negative")
        }
    }
}

@Test(.enabled(if: textTokenizer != nil, "needs the Breeze-TTS-2 tokenizer.json")) func textPromptRejectsWhatTheReferenceRejects() throws {
    let tokenizer = try #require(textTokenizer)
    let voice = try BreezeReferenceVoice(transcript: "Hi.", codes: [[Int16]](repeating: [Int16](repeating: 0, count: 16), count: 3))
    let design = try BreezeRequest(text: "Hello.", task: .voiceDesign(instruction: "Warm."))
    for guidance: Float in [0, -1, .nan, .infinity] {
        #expect(throws: BreezeTTSError.self) { try BreezePrompt(fragment: "Hello.", of: design, guidance: guidance, tokenizer: tokenizer) }
    }
    for request in [try BreezeRequest(text: "Hello."), try BreezeRequest(text: "Hello.", task: .voiceClone(voice))] {
        #expect(throws: BreezeTTSError.self) { try BreezePrompt(fragment: "Hello.", of: request, guidance: 4, tokenizer: tokenizer) }
        #expect(try BreezePrompt(fragment: "Hello.", of: request, guidance: 1, tokenizer: tokenizer).guidance == nil)
    }
    let long = try BreezeReferenceVoice(transcript: "Hi.", codes: [[Int16]](repeating: [Int16](repeating: 0, count: 16), count: 2040))
    let clone = try BreezeRequest(text: "Hello there.", task: .voiceClone(long))
    #expect(throws: BreezeTTSError.self) { try BreezePrompt(fragment: clone.text, of: clone, guidance: 1, tokenizer: tokenizer) }
}

/// The decoder in the reference's render step maps U+2581 to a space, so `" ▁▁ "` must encode as
/// four spaces, not as a space, the `▁▁` added token and a space.
@Test(.enabled(if: textTokenizer != nil, "needs the Breeze-TTS-2 tokenizer.json")) func textPromptRendersLowLinesAsSpaces() throws {
    let tokenizer = try #require(textTokenizer)
    let request = try BreezeRequest(text: "a ▁▁ b")
    let prompt = try BreezePrompt(fragment: request.text, of: request, guidance: 1, tokenizer: tokenizer)
    #expect(prompt.positive.ids == [2, 262146] + tokenizer.encode("a    b"))
    #expect(prompt.positive.ids != [2, 262146] + tokenizer.encode("a ▁▁ b"))
}
