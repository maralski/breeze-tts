// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The checkpoint's Gemma 3 tokenizer, read from tokenizer.json. `encode` matches Hugging Face
/// `tokenizer(text, add_special_tokens=False)`: added tokens split first (leftmost-longest on the
/// raw text), each gap normalized (space to U+2581) and run through BPE by merge rank as one word,
/// with characters missing from the vocab spelled as `<0xNN>` byte tokens.
struct BreezeTokenizer: Sendable {
    let bosID: Int32

    private struct Merge {
        var rank: Int32
        var id: Int32
    }

    private let vocab: ByteTable
    private let byteIDs: [Int32?]
    private let unkID: Int32?
    private let fuseUnknown: Bool
    private let merges: [UInt64: Merge]
    private let added: AddedTokenTrie

    init(tokenizerJSON url: URL) throws {
        let data = try Data(contentsOf: url)
        func fail(_ what: String) -> BreezeTTSError { .badCheckpoint("\(url.lastPathComponent): \(what)") }
        var config: [String: JSONScanner.Value] = [:]
        var vocab = ByteTable(capacity: 1 << 19)
        var merges: [UInt64: Merge] = [:]
        var addedTokens: [JSONScanner.Value] = []
        do {
            try data.withUnsafeBytes { raw in
                var scanner = JSONScanner(raw.bindMemory(to: UInt8.self))
                try scanner.forEachField { scanner, key in
                    switch String(decoding: key, as: UTF8.self) {
                    case "model":
                        try scanner.forEachField { scanner, key in
                            switch String(decoding: key, as: UTF8.self) {
                            case "vocab": try Self.readVocab(&scanner, into: &vocab)
                            case "merges":
                                guard vocab.count > 0 else { throw fail("merges come before the vocab") }
                                merges = try Self.readMerges(&scanner, vocab: vocab, fail: fail)
                            case let other: config["model." + other] = try scanner.readValue()
                            }
                        }
                    case "added_tokens": addedTokens = try scanner.readValue().array ?? []
                    case let other: config[other] = try scanner.readValue()
                    }
                }
            }
        } catch let error as JSONScanner.Malformed {
            throw fail("malformed JSON at byte \(error.offset)")
        }

        typealias V = JSONScanner.Value
        let expected: [String: V] = [
            "model.type": .string("BPE"), "model.byte_fallback": .bool(true), "model.dropout": .null,
            "model.continuing_subword_prefix": .null, "model.end_of_word_suffix": .null, "model.ignore_merges": .bool(false),
            "normalizer": .object(["type": .string("Replace"), "pattern": .object(["String": .string(" ")]), "content": .string("\u{2581}")]),
        ]
        for (key, value) in expected where config[key] != value { throw fail("\(key) is not \(value)") }
        // The pre-tokenizer splits on U+0020, which the normalizer has already replaced: a no-op.
        let split: V = .object(["type": .string("Split"), "pattern": .object(["String": .string(" ")]),
                                "behavior": .string("MergedWithPrevious"), "invert": .bool(false)])
        guard config["pre_tokenizer"] == split || config["pre_tokenizer"] == .null else { throw fail("unsupported pre_tokenizer") }

        let post = config["post_processor"]
        guard let single = post?["single"]?.array, single.count == 2,
              let bosName = single[0]["SpecialToken"]?["id"]?.string, single[1]["Sequence"]?["id"]?.string == "A",
              let bos = post?["special_tokens"]?[bosName]?["ids"]?.array?.first?.int
        else { throw fail("post_processor is not <bos> + sequence") }
        bosID = Int32(bos)

        var trie = AddedTokenTrie()
        for token in addedTokens {
            guard let content = token["content"]?.string, let id = token["id"]?.int else { throw fail("added token without content or id") }
            for flag in ["lstrip", "rstrip", "single_word", "normalized"] where token[flag]?.bool != false {
                throw fail("added token \(content) sets \(flag), which this tokenizer does not implement")
            }
            trie.insert(Array(content.utf8), id: Int32(id))
        }
        added = trie

        byteIDs = (0..<256).map { byte in vocab.id(of: Array(String(format: "<0x%02X>", byte).utf8)) }
        unkID = config["model.unk_token"]?.string.flatMap { vocab.id(of: Array($0.utf8)) }
        fuseUnknown = config["model.fuse_unk"]?.bool ?? false
        self.vocab = vocab
        self.merges = merges
    }

    private static func readVocab(_ scanner: inout JSONScanner, into vocab: inout ByteTable) throws {
        try scanner.forEachField { scanner, key in
            vocab.insert(key, id: Int32(try scanner.readInt()))
        }
    }

    /// Later duplicates of a pair overwrite earlier ones, as in `tokenizers`.
    private static func readMerges(_ scanner: inout JSONScanner, vocab: ByteTable,
                                   fail: (String) -> BreezeTTSError) throws -> [UInt64: Merge] {
        var merges: [UInt64: Merge] = [:]
        merges.reserveCapacity(1 << 20)
        var left: [UInt8] = [], right: [UInt8] = [], joined: [UInt8] = []
        var rank: Int32 = 0
        try scanner.forEachElement { scanner in
            left.removeAll(keepingCapacity: true)
            right.removeAll(keepingCapacity: true)
            if try scanner.isString() {
                try scanner.readString(into: &left)
                guard let space = left.firstIndex(of: 0x20), !left[(space + 1)...].contains(0x20) else { throw fail("merge \(rank)") }
                right.append(contentsOf: left[(space + 1)...])
                left.removeSubrange(space...)
            } else {
                var parts = 0
                try scanner.forEachElement { scanner in
                    parts += 1
                    try scanner.readString(into: &(parts == 1 ? left : right))
                }
                guard parts == 2 else { throw fail("merge \(rank)") }
            }
            joined.removeAll(keepingCapacity: true)
            joined.append(contentsOf: left)
            joined.append(contentsOf: right)
            guard let a = vocab.id(of: left), let b = vocab.id(of: right), let merged = vocab.id(of: joined)
            else { throw fail("merge \(rank) names a token outside the vocab") }
            merges[key(a, b)] = Merge(rank: rank, id: merged)
            rank += 1
        }
        return merges
    }

    /// The id of an added token such as `<|AUDIO|>`, or nil when the tokenizer has none by that name.
    func addedTokenID(_ content: String) -> Int32? {
        let bytes = Array(content.utf8)
        guard let match = added.longestMatch(bytes, at: 0), match.length == bytes.count else { return nil }
        return match.id
    }

    func encode(_ text: String) -> [Int32] {
        let input = Array(text.utf8)
        var ids: [Int32] = []
        ids.reserveCapacity(input.count / 3)
        var start = 0, i = 0
        while i < input.count {
            if let match = added.longestMatch(input, at: i) {
                if start < i { encodeWord(input[start..<i], into: &ids) }
                ids.append(match.id)
                i += match.length
                start = i
            } else {
                i += 1
            }
        }
        if start < input.count { encodeWord(input[start...], into: &ids) }
        return ids
    }

    /// `BPE::merge_word` then `Word::merge_all` from `tokenizers`, on one normalized gap.
    private func encodeWord(_ raw: ArraySlice<UInt8>, into ids: inout [Int32]) {
        var word: [UInt8] = []
        word.reserveCapacity(raw.count + raw.count / 2)
        for byte in raw {
            if byte == 0x20 { word.append(contentsOf: [0xE2, 0x96, 0x81]) } else { word.append(byte) }
        }

        var symbols: [Int32] = []
        symbols.reserveCapacity(word.count)
        var unknown = false
        word.withUnsafeBufferPointer { word in
            var i = 0
            while i < word.count {
                let length = utf8Length(word[i])
                let char = UnsafeBufferPointer(rebasing: word[i..<min(i + length, word.count)])
                i += length
                if let id = vocab.id(of: char) {
                    if unknown, let unkID { symbols.append(unkID); unknown = false }
                    symbols.append(id)
                    continue
                }
                let fallback = char.compactMap { byteIDs[Int($0)] }
                if fallback.count == char.count {
                    symbols.append(contentsOf: fallback)
                    continue
                }
                guard let unkID else { continue }
                if unknown && !fuseUnknown { symbols.append(unkID) }
                unknown = true
            }
        }
        if unknown, let unkID { symbols.append(unkID) }
        mergeAll(&symbols)
        ids.append(contentsOf: symbols)
    }

    /// Lowest rank first, then leftmost. A queued merge goes stale when either side has changed; it
    /// is skipped only if the pair now at its position maps to a different merged id, as upstream.
    private func mergeAll(_ symbols: inout [Int32]) {
        let n = symbols.count
        guard n > 1 else { return }
        var prev = (0..<n).map { Int32($0) - 1 }
        var next = (0..<n).map { $0 + 1 < n ? Int32($0 + 1) : -1 }
        var alive = [Bool](repeating: true, count: n)
        var queue = MergeQueue()
        for pos in 0..<(n - 1) {
            if let merge = merges[Self.key(symbols[pos], symbols[pos + 1])] { queue.push(rank: merge.rank, pos: Int32(pos), id: merge.id) }
        }
        while let top = queue.pop() {
            let pos = Int(top.pos)
            guard alive[pos], next[pos] >= 0 else { continue }
            let right = Int(next[pos])
            guard let current = merges[Self.key(symbols[pos], symbols[right])], current.id == top.id else { continue }
            symbols[pos] = top.id
            alive[right] = false
            next[pos] = next[right]
            if next[right] >= 0 { prev[Int(next[right])] = Int32(pos) }
            if prev[pos] >= 0, let merge = merges[Self.key(symbols[Int(prev[pos])], symbols[pos])] {
                queue.push(rank: merge.rank, pos: prev[pos], id: merge.id)
            }
            if next[pos] >= 0, let merge = merges[Self.key(symbols[pos], symbols[Int(next[pos])])] {
                queue.push(rank: merge.rank, pos: Int32(pos), id: merge.id)
            }
        }
        symbols = zip(symbols, alive).compactMap { $1 ? $0 : nil }
    }

    private static func key(_ a: Int32, _ b: Int32) -> UInt64 { UInt64(UInt32(a)) << 32 | UInt64(UInt32(b)) }
}

private func utf8Length(_ lead: UInt8) -> Int {
    switch lead {
    case 0xF0...: 4
    case 0xE0...: 3
    case 0xC0...: 2
    default: 1
    }
}

/// Vocab strings keyed by their UTF-8 bytes. Swift `String` keys would compare by canonical
/// equivalence, and 434 vocab entries collide under NFC.
private struct ByteTable: Sendable {
    private var arena: [UInt8] = []
    private var starts: [Int32] = [0]
    private var ids: [Int32] = []
    private var slots: [Int32]
    private let mask: Int
    var count: Int { ids.count }

    init(capacity: Int) {
        slots = [Int32](repeating: -1, count: capacity)
        mask = capacity - 1
        arena.reserveCapacity(capacity * 8)
    }

    mutating func insert(_ bytes: [UInt8], id: Int32) {
        precondition(count * 2 < slots.count, "ByteTable over half full")
        var slot = bytes.withUnsafeBufferPointer(Self.hash) & mask
        while slots[slot] >= 0 {
            let entry = Int(slots[slot])
            if bytes.withUnsafeBufferPointer({ equals(entry, $0) }) { ids[entry] = id; return }
            slot = (slot + 1) & mask
        }
        slots[slot] = Int32(ids.count)
        arena.append(contentsOf: bytes)
        starts.append(Int32(arena.count))
        ids.append(id)
    }

    func id(of bytes: [UInt8]) -> Int32? { bytes.withUnsafeBufferPointer { id(of: $0) } }

    func id(of bytes: UnsafeBufferPointer<UInt8>) -> Int32? {
        var slot = Self.hash(bytes) & mask
        while slots[slot] >= 0 {
            let entry = Int(slots[slot])
            if equals(entry, bytes) { return ids[entry] }
            slot = (slot + 1) & mask
        }
        return nil
    }

    private func equals(_ entry: Int, _ bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        let start = Int(starts[entry]), end = Int(starts[entry + 1])
        guard end - start == bytes.count else { return false }
        return arena.withUnsafeBufferPointer { arena in
            for k in 0..<bytes.count where arena[start + k] != bytes[k] { return false }
            return true
        }
    }

    private static func hash(_ bytes: UnsafeBufferPointer<UInt8>) -> Int {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in bytes { h = (h ^ UInt64(byte)) &* 0x100_0000_01b3 }
        return Int(truncatingIfNeeded: h ^ (h >> 29))
    }
}

/// Byte trie over the added tokens for leftmost-longest matching, as the `tokenizers`
/// Aho-Corasick automaton (`MatchKind::LeftmostLongest`) does.
private struct AddedTokenTrie: Sendable {
    private var children: [UInt32: Int32] = [:]
    private var terminal: [Int32] = [-1]

    mutating func insert(_ bytes: [UInt8], id: Int32) {
        var node: Int32 = 0
        for byte in bytes {
            let edge = UInt32(node) << 8 | UInt32(byte)
            if let child = children[edge] {
                node = child
            } else {
                terminal.append(-1)
                node = Int32(terminal.count - 1)
                children[edge] = node
            }
        }
        terminal[Int(node)] = id
    }

    func longestMatch(_ bytes: [UInt8], at start: Int) -> (id: Int32, length: Int)? {
        var node: Int32 = 0, best: (id: Int32, length: Int)?
        var i = start
        while i < bytes.count, let child = children[UInt32(node) << 8 | UInt32(bytes[i])] {
            node = child
            i += 1
            if terminal[Int(node)] >= 0 { best = (terminal[Int(node)], i - start) }
        }
        return best
    }
}

/// Binary min-heap ordered by (rank, position), the order of `tokenizers`' `Merge`.
private struct MergeQueue {
    private var items: [(rank: Int32, pos: Int32, id: Int32)] = []

    private static func less(_ a: (rank: Int32, pos: Int32, id: Int32), _ b: (rank: Int32, pos: Int32, id: Int32)) -> Bool {
        a.rank != b.rank ? a.rank < b.rank : a.pos < b.pos
    }

    mutating func push(rank: Int32, pos: Int32, id: Int32) {
        items.append((rank, pos, id))
        var child = items.count - 1
        while child > 0 {
            let parent = (child - 1) / 2
            guard Self.less(items[child], items[parent]) else { break }
            items.swapAt(child, parent)
            child = parent
        }
    }

    mutating func pop() -> (rank: Int32, pos: Int32, id: Int32)? {
        guard let first = items.first else { return nil }
        let last = items.removeLast()
        if !items.isEmpty {
            items[0] = last
            var parent = 0
            while true {
                let left = 2 * parent + 1, right = left + 1
                var smallest = parent
                if left < items.count && Self.less(items[left], items[smallest]) { smallest = left }
                if right < items.count && Self.less(items[right], items[smallest]) { smallest = right }
                if smallest == parent { break }
                items.swapAt(parent, smallest)
                parent = smallest
            }
        }
        return first
    }
}
