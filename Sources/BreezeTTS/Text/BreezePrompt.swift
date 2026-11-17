// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// The backbone inputs for one fragment: the conditional branch and, under guidance, the
/// unconditional one. Each branch is a list of segments; the text encoder runs on each text
/// segment alone, and the backbone sees the segments' rows end to end.
public struct BreezePrompt: Sendable, Equatable {
    public static let audioTokenID: Int32 = 262144
    public static let audioEndTokenID: Int32 = 262145
    /// Positions in the backbone cache, shared by the prompt and the generated frames.
    public static let backbonePositions = 2048

    public enum Segment: Sendable, Equatable {
        /// Token ids, starting with `<bos>`.
        case text([Int32])
        /// A reference clip's codes, one row of 16 per frame. The layout follows it with one
        /// end-of-audio row, embedded as a frame of all-zero codes.
        case audio([[Int16]])
    }

    public enum Row: Sendable, Equatable {
        /// The next row of the concatenated, projected text-encoder outputs.
        case text
        /// The next frame of the concatenated reference codes.
        case audio
        case audioEnd
    }

    public struct Branch: Sendable, Equatable {
        public let segments: [Segment]

        public var rows: [Row] {
            segments.flatMap { segment -> [Row] in
                switch segment {
                case .text(let ids): Array(repeating: .text, count: ids.count)
                case .audio(let codes): Array(repeating: .audio, count: codes.count) + [.audioEnd]
                }
            }
        }

        /// The reference's backbone `input_ids`.
        public var ids: [Int32] {
            segments.flatMap { segment -> [Int32] in
                switch segment {
                case .text(let ids): ids
                case .audio(let codes): Array(repeating: BreezePrompt.audioTokenID, count: codes.count) + [BreezePrompt.audioEndTokenID]
                }
            }
        }

        public var count: Int {
            segments.reduce(0) { total, segment in
                switch segment {
                case .text(let ids): total + ids.count
                case .audio(let codes): total + codes.count + 1
                }
            }
        }
    }

    public struct Guidance: Sendable, Equatable {
        /// `uncond + scale * (cond - uncond)`, on the backbone and every depth-decoder codebook.
        public let scale: Float
        public let negative: Branch
    }

    public let positive: Branch
    /// Nil at guidance 1, where the reference runs the positive branch alone.
    public let guidance: Guidance?
}

extension BreezePrompt {
    /// `templates.prepare_inputs` for `fragment` under `request`'s task. Throws where the
    /// reference raises: guidance other than 1 for a task without a negative prompt, guidance
    /// that is not a positive number, and a prompt that leaves the backbone no room to generate.
    init(fragment: String, of request: BreezeRequest, guidance: Float, tokenizer: BreezeTokenizer) throws {
        guard guidance.isFinite, guidance > 0 else { throw BreezeTTSError.badInput("guidance must be a positive number, not \(guidance)") }
        guard guidance == 1 || request.task.acceptsGuidance else {
            throw BreezeTTSError.badInput("plain and clone requests take no guidance (got \(guidance)); they have no negative prompt")
        }
        func text(_ s: String) -> Segment { .text(Self.encodeSegment(s, tokenizer)) }
        func clone(_ voice: BreezeReferenceVoice) -> [Segment] {
            [text("[S0]" + voice.transcript), .audio(voice.codes), text("[S0]" + fragment)]
        }
        let directed = { (instruction: String) in text("[S0]<ins_bos>" + instruction + "<ins_eos>" + fragment) }
        let positive: [Segment], negative: [Segment]?
        switch request.task {
        case .plain:
            (positive, negative) = ([text("[S0]" + fragment)], nil)
        case .voiceDesign(let instruction):
            (positive, negative) = ([directed(instruction)], [text("[S0]" + fragment)])
        case .voiceClone(let voice):
            (positive, negative) = (clone(voice), nil)
        case .voiceDirection(let voice, let instruction):
            (positive, negative) = (Array(clone(voice).dropLast()) + [directed(instruction)], clone(voice))
        }
        self.positive = Branch(segments: positive)
        self.guidance = guidance == 1 ? nil : negative.map { Guidance(scale: guidance, negative: Branch(segments: $0)) }
        for branch in [self.positive, self.guidance?.negative].compactMap({ $0 }) where branch.count >= Self.backbonePositions - 1 {
            throw BreezeTTSError.badInput("the prompt fills \(branch.count) of the backbone's \(Self.backbonePositions) positions; shorten the text or the reference clip")
        }
    }

    /// The reference renders a text segment as `decode(encode(text))` before the joint encode, and
    /// the decoder turns every U+2581 into a space. A literal `▁▁` is an added token; its decoded
    /// `"  "` is not, so encoding the raw text would differ.
    static func encodeSegment(_ text: String, _ tokenizer: BreezeTokenizer) -> [Int32] {
        let rendered = String(String.UnicodeScalarView(text.unicodeScalars.map { $0 == "\u{2581}" ? " " : $0 }))
        return [tokenizer.bosID] + tokenizer.encode(rendered)
    }
}
