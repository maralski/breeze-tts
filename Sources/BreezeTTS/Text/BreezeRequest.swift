// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The inline sounds the model card documents, performed where the tag is written: English text
/// takes the parenthesized form, as in `"(sigh) It is good to hear you."`, and Chinese text the
/// bracketed one, as in `"[叹气] 没想到过了这么久。"`. The card lists these four "such as"; other
/// tags reach the model as written, with no promise it performs them.
public enum BreezeVocalEvent: CaseIterable, Sendable {
    case laugh, cough, clearThroat, sigh

    public var english: String {
        switch self {
        case .laugh: "(laugh)"
        case .cough: "(cough)"
        case .clearThroat: "(clears throat)"
        case .sigh: "(sigh)"
        }
    }

    public var chinese: String {
        switch self {
        case .laugh: "[笑]"
        case .cough: "[咳嗽]"
        case .clearThroat: "[清嗓子]"
        case .sigh: "[叹气]"
        }
    }
}

/// An encoded reference clip and what it says: the voice a clone or direction request copies.
/// Encode a clip once, persist it (it is `Codable`), and reuse it for every request.
public struct BreezeReferenceVoice: Sendable, Codable, Equatable {
    public static let codebooks = 16
    public static let codebookSize = 2048
    /// One frame is 1920 samples at 24 kHz.
    public static let framesPerSecond = 12.5

    /// What the clip says, exactly, with surrounding whitespace stripped as the reference does.
    public let transcript: String
    /// The codec codes of the clip, one row of 16 per frame.
    public let codes: [[Int16]]

    public init(transcript: String, codes: [[Int16]]) throws {
        let transcript = pythonStrip(transcript)
        guard !transcript.isEmpty else { throw BreezeTTSError.badInput("a reference voice needs its transcript") }
        try rejectPromptMarkers(transcript, "the transcript")
        guard !codes.isEmpty else { throw BreezeTTSError.badInput("a reference voice needs at least one frame of codes") }
        guard codes.allSatisfy({ $0.count == Self.codebooks }) else { throw BreezeTTSError.badInput("reference codes must have \(Self.codebooks) per frame") }
        guard codes.allSatisfy({ $0.allSatisfy { (0..<Int16(Self.codebookSize)).contains($0) } })
        else { throw BreezeTTSError.badInput("reference codes must lie in 0..<\(Self.codebookSize)") }
        self.transcript = transcript
        self.codes = codes
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(transcript: container.decode(String.self, forKey: .transcript), codes: container.decode([[Int16]].self, forKey: .codes))
    }

    public var frames: Int { codes.count }
    public var seconds: Double { Double(frames) / Self.framesPerSecond }

    /// The longest clip to clone from. The clip shares the backbone's 2048 positions with the
    /// prompt text and the generated frames. 20 s is 250 frames, plus the end-of-audio row, about
    /// 80 tokens of transcript and 4 marker tokens: about 335 positions, which leaves the full
    /// 1500-frame generation cap reachable (335 + 1500 < 2047) with about 200 tokens to spare for
    /// the text. A 40 s clip costs about 670 positions and cuts the cap to about 1370 frames.
    public static let recommendedMaxSeconds = 20.0
}

/// One of the reference's four templates, chosen by which fields it carries.
public enum BreezeTask: Sendable, Equatable {
    /// `tts_plain`: the model's own voice.
    case plain
    /// `tts_instruction`: a voice made from a natural-language description. Match the
    /// instruction's language to the text.
    case voiceDesign(instruction: String)
    /// `ref_clone_tata`: the reference voice reading the text. Takes no guidance.
    case voiceClone(BreezeReferenceVoice)
    /// `ref_edit_tata`: the reference voice, directed by the instruction (tone, emotion, pace).
    case voiceDirection(BreezeReferenceVoice, instruction: String)

    /// Whether the template defines the negative prompt that guidance other than 1 needs.
    public var acceptsGuidance: Bool {
        switch self {
        case .plain, .voiceClone: false
        case .voiceDesign, .voiceDirection: true
        }
    }

    /// The upstream README's `--cfg-scale 4` for design and direction; 1 (no guidance) otherwise,
    /// the only value plain and clone accept. The reference CLI defaults to 1 for every task.
    public var recommendedGuidance: Float { acceptsGuidance ? 4 : 1 }

    public var reference: BreezeReferenceVoice? {
        switch self {
        case .plain, .voiceDesign: nil
        case .voiceClone(let voice), .voiceDirection(let voice, _): voice
        }
    }
}

/// Everything one synthesis needs. The initializer rejects what the reference rejects, and
/// guidance is checked where the prompt is built.
public struct BreezeRequest: Sendable, Equatable {
    /// As given: the reference does not strip it, and leading spaces change the tokens.
    public let text: String
    /// With the instruction stripped, as the reference does.
    public let task: BreezeTask

    public init(text: String, task: BreezeTask = .plain) throws {
        guard !pythonStrip(text).isEmpty else { throw BreezeTTSError.badInput("text must not be empty") }
        try rejectPromptMarkers(text, "the text")
        switch task {
        case .voiceDesign(let instruction):
            self.task = .voiceDesign(instruction: try Self.instruction(instruction))
        case .voiceDirection(let voice, let instruction):
            self.task = .voiceDirection(voice, instruction: try Self.instruction(instruction))
        case .plain, .voiceClone:
            self.task = task
        }
        self.text = text
    }

    /// The reference reads a blank instruction as none and silently switches template; a typed
    /// task that asks for one must carry one.
    private static func instruction(_ raw: String) throws -> String {
        let instruction = pythonStrip(raw)
        guard !instruction.isEmpty else { throw BreezeTTSError.badInput("the task needs a non-empty instruction") }
        try rejectPromptMarkers(instruction, "the instruction")
        return instruction
    }
}

/// The backbone finds audio rows by these token ids, so text holding them would take codec
/// embeddings meant for the reference clip.
/// Matched on bytes, as the tokenizer matches: a combining mark after `>` hides it from `String.contains`.
private func rejectPromptMarkers(_ text: String, _ what: String) throws {
    let bytes = Array(text.utf8)
    for marker in ["<|AUDIO|>", "<|audio_eos|>"] where bytes.indices.contains(where: { bytes[$0...].starts(with: marker.utf8) }) {
        throw BreezeTTSError.badInput("\(what) must not contain \(marker)")
    }
}

/// Python `str.strip()`: Unicode White_Space plus the ASCII separators U+001C to U+001F.
func pythonStrip(_ text: String) -> String {
    let s = text.unicodeScalars
    guard let first = s.firstIndex(where: { !isPythonSpace($0) }), let last = s.lastIndex(where: { !isPythonSpace($0) })
    else { return "" }
    return String(s[first...last])
}

func isPythonSpace(_ c: Unicode.Scalar) -> Bool {
    c.properties.isWhitespace || (0x1C...0x1F).contains(c.value)
}
