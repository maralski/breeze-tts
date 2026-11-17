// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

extension BreezeRequest {
    /// The default `fragments(maxLength:)` budget, in words and CJK characters. An estimate, not
    /// a measurement: 60 English words at a slow 2 words/s is 30 s, 375 frames, and 60 Chinese
    /// characters at 4/s is 15 s; either sits well inside the 1500-frame cap and costs under 100
    /// prompt tokens, which leaves the backbone room for a 20 s reference clip.
    public static let defaultFragmentLength = 60

    /// The text cut into fragments of at most `maxLength` words and CJK characters, to synthesize
    /// one at a time. Text within the budget comes back unchanged. Longer text is cut at sentence
    /// ends where it can, then at clause punctuation, then between words; never inside a
    /// vocal-event tag, a URL, an email address, a number or an abbreviation. A single word or
    /// tag longer than the budget becomes a fragment of its own. The reference does no splitting
    /// (it caps generation at 1500 frames); this is the port's own.
    public func fragments(maxLength: Int = defaultFragmentLength) -> [String] {
        splitText(text, maxLength: maxLength)
    }
}

func splitText(_ text: String, maxLength: Int) -> [String] {
    precondition(maxLength >= 1, "maxLength must be positive")
    let s = Array(text.unicodeScalars)
    let unitsBefore = countUnits(s)
    guard unitsBefore[s.count] > maxLength else { return [text] }
    let span = protectedSpans(text, count: s.count)

    var ranges: [Range<Int>] = []
    var start = 0
    while start < s.count {
        var end = s.count
        if unitsBefore[s.count] - unitsBefore[start] > maxLength {
            var best: [Int?] = [nil, nil, nil]
            var firstCut: Int?
            var b = start + 1
            while b < s.count {
                if let level = cutLevel(s, at: b, span: span) {
                    if unitsBefore[b] - unitsBefore[start] > maxLength { firstCut = firstCut ?? b; break }
                    best[level] = b
                    firstCut = firstCut ?? b
                }
                b += 1
            }
            end = best[2] ?? best[1] ?? best[0] ?? firstCut ?? s.count
        }
        if unitsBefore[end] == unitsBefore[start], let last = ranges.indices.last {
            ranges[last] = ranges[last].lowerBound..<end
        } else {
            ranges.append(start..<end)
        }
        start = end
    }
    return ranges.map { pythonStrip(String(String.UnicodeScalarView(s[$0]))) }.filter { !$0.isEmpty }
}

/// `unitsBefore[i]`: words and CJK characters that start before scalar `i`. A word is a run of
/// letters, digits and marks, joined across an apostrophe (`don't`) or a separator between digits
/// (`3.50`, `9:30`, `1,000`).
private func countUnits(_ s: [Unicode.Scalar]) -> [Int] {
    var units = [Int](repeating: 0, count: s.count + 1)
    var inWord = false
    func isDigit(_ c: Unicode.Scalar) -> Bool { c.properties.generalCategory == .decimalNumber }
    for i in s.indices {
        let c = s[i]
        var starts = false
        let next = i + 1 < s.count ? s[i + 1] : nil
        if isCJK(c) {
            starts = true
            inWord = false
        } else if isWordScalar(c) {
            starts = !inWord
            inWord = true
        } else if inWord, let next, ((c == "'" || c == "\u{2019}") && isWordScalar(next) && !isCJK(next))
                    || (".,:".unicodeScalars.contains(c) && isDigit(s[i - 1]) && isDigit(next)) {
            inWord = true
        } else {
            inWord = false
        }
        units[i + 1] = units[i] + (starts ? 1 : 0)
    }
    return units
}

private func isWordScalar(_ c: Unicode.Scalar) -> Bool {
    switch c.properties.generalCategory {
    case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
         .decimalNumber, .letterNumber, .otherNumber, .nonspacingMark, .spacingMark, .enclosingMark: true
    default: false
    }
}

/// Han, kana and Hangul syllables: each is one unit, and a cut may fall between any two.
private func isCJK(_ c: Unicode.Scalar) -> Bool {
    switch c.value {
    case 0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF, 0x20000...0x3134F,
         0x3041...0x3096, 0x309D...0x309F, 0x30A1...0x30FA, 0x30FC...0x30FF, 0xAC00...0xD7A3: true
    default: false
    }
}

private let sentenceEnds = Set("。！？!?.…".unicodeScalars)
private let asciiEnds = Set("!?.,;:".unicodeScalars)
private let clauseEnds = Set("，、；：,;:—".unicodeScalars)
private let closers = Set("\"'”’」』）)]】》".unicodeScalars)
private let punctuation = sentenceEnds.union(clauseEnds).union(closers)

/// 2 for a sentence end or line break, 1 for clause punctuation, 0 between words; nil where a
/// fragment must not end before scalar `b`. ASCII punctuation counts only before whitespace,
/// so `3.5`, `a,b` and `node.js` hold together.
private func cutLevel(_ s: [Unicode.Scalar], at b: Int, span: [Int]) -> Int? {
    if span[b - 1] >= 0 && span[b - 1] == span[b] { return nil }
    let p = s[b - 1], q = s[b]
    if p == "\n" { return 2 }
    if span[b - 1] < 0, punctuation.contains(p), !punctuation.contains(q) || span[b] >= 0 {
        var i = b - 1, level: Int?
        while i >= 0, span[i] < 0, punctuation.contains(s[i]) {
            let c = s[i]
            if !asciiEnds.contains(c) || q.properties.isWhitespace {
                if sentenceEnds.contains(c) { level = 2 } else if clauseEnds.contains(c) { level = max(level ?? 1, 1) }
            }
            i -= 1
        }
        if let level { return level }
    }
    if q.properties.isWhitespace && !p.properties.isWhitespace { return 0 }
    if (isCJK(p) || isCJK(q)) && !p.properties.isWhitespace && !q.properties.isWhitespace { return 0 }
    return nil
}

/// For each scalar, the index of the protected span holding it, or -1.
private func protectedSpans(_ text: String, count: Int) -> [Int] {
    var span = [Int](repeating: -1, count: count + 1)
    var scalarAt = [Int](), index = 0
    for scalar in text.unicodeScalars {
        scalarAt.append(contentsOf: repeatElement(index, count: scalar.utf16.count))
        index += 1
    }
    scalarAt.append(index)
    let ns = text as NSString
    for (n, match) in protectedPattern.matches(in: text, range: NSRange(location: 0, length: ns.length)).enumerated() {
        for i in scalarAt[match.range.lowerBound]..<scalarAt[match.range.upperBound] { span[i] = n }
    }
    return span
}

private let protectedPattern = try! NSRegularExpression(pattern: [
    #"\([^()\[\]\n]{1,32}\)"#, #"\[[^()\[\]\n]{1,32}\]"#,
    #"(?i:https?://|www\.)[^\s，。！？]+"#,
    #"[\p{L}\p{N}_.+-]+@[\p{L}\p{N}_-]+(?:\.[\p{L}\p{N}_-]+)+"#,
    #"\d+(?:[.,:]\d+)+"#, #"(?:[A-Za-z]\.){2,}"#,
    #"(?i:(?<![A-Za-z])(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|No|Fig|approx)\.)"#,
].joined(separator: "|"))
