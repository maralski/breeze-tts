// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A pull parser over UTF-8 JSON bytes. tokenizer.json is 33 MB, almost all of it the vocab and
/// merge lists; streaming those straight into byte-keyed tables avoids the 0.6 s that
/// JSONSerialization and its bridging casts spend building 780k Foundation objects.
struct JSONScanner {
    struct Malformed: Error { var offset: Int }

    indirect enum Value: Equatable {
        case null, bool(Bool), number(Double), string(String), array([Value]), object([String: Value])

        subscript(key: String) -> Value? {
            if case .object(let fields) = self { fields[key] } else { nil }
        }
        var string: String? { if case .string(let s) = self { s } else { nil } }
        var bool: Bool? { if case .bool(let b) = self { b } else { nil } }
        var int: Int? { if case .number(let n) = self { Int(exactly: n) } else { nil } }
        var array: [Value]? { if case .array(let a) = self { a } else { nil } }
    }

    private let bytes: UnsafeBufferPointer<UInt8>
    private(set) var offset = 0

    init(_ bytes: UnsafeBufferPointer<UInt8>) { self.bytes = bytes }

    private mutating func skipWhitespace() {
        while offset < bytes.count {
            switch bytes[offset] {
            case 0x20, 0x0A, 0x0D, 0x09: offset += 1
            default: return
            }
        }
    }

    private mutating func peek() throws -> UInt8 {
        skipWhitespace()
        guard offset < bytes.count else { throw Malformed(offset: offset) }
        return bytes[offset]
    }

    private mutating func expect(_ byte: UInt8) throws {
        guard try peek() == byte else { throw Malformed(offset: offset) }
        offset += 1
    }

    /// Calls `body` once per key with the scanner positioned at the value, which `body` must consume.
    mutating func forEachField(_ body: (inout JSONScanner, [UInt8]) throws -> Void) throws {
        try expect(UInt8(ascii: "{"))
        if try peek() == UInt8(ascii: "}") { offset += 1; return }
        var key: [UInt8] = []
        while true {
            key.removeAll(keepingCapacity: true)
            try readString(into: &key)
            try expect(UInt8(ascii: ":"))
            try body(&self, key)
            switch try peek() {
            case UInt8(ascii: ","): offset += 1
            case UInt8(ascii: "}"): offset += 1; return
            default: throw Malformed(offset: offset)
            }
        }
    }

    /// Calls `body` once per element with the scanner positioned at it, which `body` must consume.
    mutating func forEachElement(_ body: (inout JSONScanner) throws -> Void) throws {
        try expect(UInt8(ascii: "["))
        if try peek() == UInt8(ascii: "]") { offset += 1; return }
        while true {
            try body(&self)
            switch try peek() {
            case UInt8(ascii: ","): offset += 1
            case UInt8(ascii: "]"): offset += 1; return
            default: throw Malformed(offset: offset)
            }
        }
    }

    mutating func isString() throws -> Bool { try peek() == UInt8(ascii: "\"") }

    /// Appends the string's UTF-8 bytes, escapes decoded, to `out`.
    mutating func readString(into out: inout [UInt8]) throws {
        try expect(UInt8(ascii: "\""))
        while offset < bytes.count {
            let byte = bytes[offset]
            offset += 1
            switch byte {
            case UInt8(ascii: "\""): return
            case UInt8(ascii: "\\"): try readEscape(into: &out)
            default: out.append(byte)
            }
        }
        throw Malformed(offset: offset)
    }

    private mutating func readEscape(into out: inout [UInt8]) throws {
        guard offset < bytes.count else { throw Malformed(offset: offset) }
        let byte = bytes[offset]
        offset += 1
        switch byte {
        case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): out.append(byte)
        case UInt8(ascii: "b"): out.append(0x08)
        case UInt8(ascii: "f"): out.append(0x0C)
        case UInt8(ascii: "n"): out.append(0x0A)
        case UInt8(ascii: "r"): out.append(0x0D)
        case UInt8(ascii: "t"): out.append(0x09)
        case UInt8(ascii: "u"):
            var unit = try readHex4()
            if (0xD800..<0xDC00).contains(unit), offset + 1 < bytes.count,
               bytes[offset] == UInt8(ascii: "\\"), bytes[offset + 1] == UInt8(ascii: "u") {
                offset += 2
                let low = try readHex4()
                guard (0xDC00..<0xE000).contains(low) else { throw Malformed(offset: offset) }
                unit = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
            }
            guard let scalar = Unicode.Scalar(unit) else { throw Malformed(offset: offset) }
            out.append(contentsOf: Array(String(scalar).utf8))
        default: throw Malformed(offset: offset)
        }
    }

    private mutating func readHex4() throws -> UInt32 {
        guard offset + 4 <= bytes.count else { throw Malformed(offset: offset) }
        var value: UInt32 = 0
        for _ in 0..<4 {
            let byte = bytes[offset]
            offset += 1
            let digit: UInt32
            switch byte {
            case UInt8(ascii: "0")...UInt8(ascii: "9"): digit = UInt32(byte - UInt8(ascii: "0"))
            case UInt8(ascii: "a")...UInt8(ascii: "f"): digit = UInt32(byte - UInt8(ascii: "a") + 10)
            case UInt8(ascii: "A")...UInt8(ascii: "F"): digit = UInt32(byte - UInt8(ascii: "A") + 10)
            default: throw Malformed(offset: offset)
            }
            value = value << 4 | digit
        }
        return value
    }

    /// A non-negative integer without fraction or exponent, the only numbers the vocab holds.
    mutating func readInt() throws -> Int {
        _ = try peek()
        var value = 0, digits = 0
        while offset < bytes.count, (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(bytes[offset]) {
            value = value * 10 + Int(bytes[offset] - UInt8(ascii: "0"))
            offset += 1
            digits += 1
        }
        guard digits > 0 else { throw Malformed(offset: offset) }
        return value
    }

    mutating func readValue() throws -> Value {
        switch try peek() {
        case UInt8(ascii: "{"):
            var fields: [String: Value] = [:]
            try forEachField { scanner, key in fields[String(decoding: key, as: UTF8.self)] = try scanner.readValue() }
            return .object(fields)
        case UInt8(ascii: "["):
            var items: [Value] = []
            try forEachElement { items.append(try $0.readValue()) }
            return .array(items)
        case UInt8(ascii: "\""):
            var out: [UInt8] = []
            try readString(into: &out)
            return .string(String(decoding: out, as: UTF8.self))
        case UInt8(ascii: "t"): try literal("true"); return .bool(true)
        case UInt8(ascii: "f"): try literal("false"); return .bool(false)
        case UInt8(ascii: "n"): try literal("null"); return .null
        default:
            let start = offset
            while offset < bytes.count, "+-.0123456789eE".utf8.contains(bytes[offset]) { offset += 1 }
            guard let number = Double(String(decoding: bytes[start..<offset], as: UTF8.self)) else { throw Malformed(offset: start) }
            return .number(number)
        }
    }

    private mutating func literal(_ word: String) throws {
        for byte in word.utf8 {
            guard offset < bytes.count, bytes[offset] == byte else { throw Malformed(offset: offset) }
            offset += 1
        }
    }
}
