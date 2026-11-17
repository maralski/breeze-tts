// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import BreezeTTS

private func fragments(_ text: String, _ maxLength: Int) -> [String] {
    try! BreezeRequest(text: text).fragments(maxLength: maxLength)
}

@Test func textSplitLeavesShortTextAlone() {
    #expect(fragments("  (laugh) Hello there.  ", 60) == ["  (laugh) Hello there.  "])
}

@Test func textSplitPrefersSentencesThenClausesThenWords() {
    #expect(fragments("First sentence here. Second one is here too. Third.", 8)
        == ["First sentence here. Second one is here too.", "Third."])
    #expect(fragments("One two three, four five six, seven eight nine.", 6) == ["One two three, four five six,", "seven eight nine."])
    #expect(fragments("one two three four five six seven", 3) == ["one two three", "four five six", "seven"])
    #expect(fragments("今天天气很好。我们去公园吧！你呢？", 8) == ["今天天气很好。", "我们去公园吧！你呢？"])
    #expect(fragments("今天天气很好，我们去公园吧", 3) == ["今天天", "气很好，", "我们去", "公园吧"])
    #expect(fragments("Line one\nline two\nline three", 4) == ["Line one\nline two", "line three"])
}

@Test func textSplitHoldsAbbreviationsNumbersAndTagsTogether() {
    #expect(fragments("Hello there Dr. Smith and friends.", 4) == ["Hello there Dr. Smith", "and friends."])
    #expect(fragments("Pay 3.50 now. It is 9:30 or so.", 3) == ["Pay 3.50 now.", "It is 9:30", "or so."])
    #expect(fragments("Well (clears throat) we need to talk about it.", 1).contains("(clears throat)"))
    #expect(fragments("[清嗓子]好的，我们开始吧。", 1).contains("[清嗓子]"))
}

/// Over the tokenizer corpus and several budgets: no text is lost or reordered and no event tag is cut.
@Test func textSplitKeepsEveryCharacterAndEveryTag() throws {
    struct Fixtures: Decodable { struct Case: Decodable { var text: String }; var cases: [Case] }
    let corpus = try JSONDecoder().decode(Fixtures.self, from: Data(contentsOf: textFixture("tokenizer_fixtures.json"))).cases.map(\.text)
        + [String(repeating: "(sigh) It is good to hear your voice again. [笑] 欢迎来到今晚的故事时间，让我们一起开始吧。", count: 12)]
    let tags = BreezeVocalEvent.allCases.flatMap { [$0.english, $0.chinese] }
    func visible(_ s: String) -> [Unicode.Scalar] { s.unicodeScalars.filter { !isPythonSpace($0) } }
    var split = 0
    for text in corpus where !pythonStrip(text).isEmpty && !text.contains("<|AUDIO|>") && !text.contains("<|audio_eos|>") {
        for maxLength in [1, 2, 5, 13, 60] {
            let parts = fragments(text, maxLength)
            if parts.count > 1 { split += 1 }
            #expect(visible(parts.joined()) == visible(text), "\(text.prefix(60).debugDescription) at \(maxLength)")
            for tag in tags {
                #expect(parts.map { $0.components(separatedBy: tag).count - 1 }.reduce(0, +) == text.components(separatedBy: tag).count - 1,
                        "\(tag) in \(text.prefix(60).debugDescription)")
            }
        }
    }
    #expect(split > 500)
}
