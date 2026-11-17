// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import BreezeTTS

/// The checkpoint's tokenizer.json: `BREEZE_WEIGHTS`, else the newest Hugging Face cache snapshot.
let textTokenizerURL: URL? = {
    let fm = FileManager.default
    if let dir = ProcessInfo.processInfo.environment["BREEZE_WEIGHTS"] {
        let url = URL(fileURLWithPath: dir).appending(path: "tokenizer.json")
        return fm.fileExists(atPath: url.path) ? url : nil
    }
    let snapshots = fm.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/hub/models--BreezeBlue--Breeze-TTS-2/snapshots")
    let dirs = (try? fm.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
    return dirs.map { $0.appending(path: "tokenizer.json") }.filter { fm.fileExists(atPath: $0.path) }
        .max { ($0.modificationDate ?? .distantPast) < ($1.modificationDate ?? .distantPast) }
}()

let textTokenizer: BreezeTokenizer? = textTokenizerURL.flatMap { try? BreezeTokenizer(tokenizerJSON: $0) }

private extension URL {
    var modificationDate: Date? { try? resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate }
}

/// Fixtures live beside this file: the package declares no test resources.
func textFixture(_ name: String) -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().appending(path: "Fixtures/text/\(name)")
}

/// Written by Fixtures/text/make_text_fixtures.py from the Hugging Face tokenizer.
private struct TokenizerFixtures: Decodable {
    struct Case: Decodable { var text: String; var ids: [Int32] }
    var cases: [Case]
}

private let isRelease: Bool = {
    #if DEBUG
    false
    #else
    true
    #endif
}()

@Test(.enabled(if: textTokenizer != nil, "needs the Breeze-TTS-2 tokenizer.json")) func textTokenizerMatchesHuggingFace() throws {
    let tokenizer = try #require(textTokenizer)
    let cases = try JSONDecoder().decode(TokenizerFixtures.self, from: Data(contentsOf: textFixture("tokenizer_fixtures.json"))).cases
    #expect(cases.count > 600)
    let failures = cases.filter { tokenizer.encode($0.text) != $0.ids }
    for failure in failures.prefix(5) {
        Issue.record("\(failure.text.debugDescription.prefix(160)): got \(tokenizer.encode(failure.text).prefix(30)) want \(failure.ids.prefix(30))")
    }
    #expect(failures.isEmpty, "\(failures.count) of \(cases.count) strings differ")
}

@Test(.enabled(if: textTokenizer != nil, "needs the Breeze-TTS-2 tokenizer.json")) func textTokenizerKnowsThePromptTokens() throws {
    let tokenizer = try #require(textTokenizer)
    #expect(tokenizer.bosID == 2)
    #expect(["<|AUDIO|>", "<|audio_eos|>", "[S0]", "[S9]", "<ins_bos>", "<ins_eos>"].map(tokenizer.addedTokenID)
        == [262144, 262145, 262146, 262155, 262156, 262157])
    #expect(tokenizer.addedTokenID("[S10]") == nil && tokenizer.addedTokenID("<|AUDIO") == nil)
    #expect(tokenizer.encode("[S0]Hello world.") == [262146, 9259, 1902, 236761])
}

/// Measured on an M3 Max in release: load 0.13 s, a 2,000-character paragraph 0.4 ms. The
/// bounds leave room for a slower machine while still catching a JSONSerialization-sized
/// regression (0.6 s before any table is built) or a quadratic merge loop.
@Test(.enabled(if: textTokenizerURL != nil && isRelease, "release build with the Breeze-TTS-2 tokenizer.json"))
func textTokenizerLoadsAndEncodesFast() throws {
    let url = try #require(textTokenizerURL)
    let clock = ContinuousClock()
    var tokenizer: BreezeTokenizer?
    let load = try clock.measure { tokenizer = try BreezeTokenizer(tokenizerJSON: url) }
    let paragraph = String(repeating: "It is good to hear your voice again after all this time, 真的很好。 ", count: 30)
    _ = tokenizer!.encode(paragraph)
    let runs = 20
    let encode = clock.measure { for _ in 0..<runs { _ = tokenizer!.encode(paragraph) } } / runs
    print("tokenizer load \(load), encode \(paragraph.count) characters \(encode)")
    #expect(load < .milliseconds(500))
    #expect(encode < .milliseconds(5))
}
