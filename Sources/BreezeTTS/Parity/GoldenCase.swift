// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// One case directory of the reference goldens (tools/reference/README.md): the manifest and its
/// raw little-endian tensors.
struct GoldenCase {
    struct Tensor {
        let values: [Float]
        let shape: [Int]
    }

    let directory: URL
    let manifest: [String: Any]

    init(directory: URL) throws {
        self.directory = directory
        let url = directory.appendingPathComponent("manifest.json")
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw BreezeTTSError.badInput("\(url.path) is not a JSON object")
        }
        manifest = json
    }

    var name: String { manifest["case"] as? String ?? directory.lastPathComponent }
    var text: String? { manifest["text"] as? String }
    var instruction: String? { manifest["instruction"] as? String }
    var guidance: Float { Float(manifest["cfg_scale"] as! Double) }
    var frameCap: Int { manifest["frame_cap"] as? Int ?? 120 }
    var transcript: String? { (manifest["ref"] as? [String: Any])?["transcript"] as? String }
    /// "cond", then "uncond" under guidance.
    var branches: [String] { guidance == 1 ? ["cond"] : ["cond", "uncond"] }

    func tokens(_ branch: String) throws -> (inputIDs: [Int32], segments: [[Int32]]) {
        guard let t = (manifest["tokens"] as? [String: Any])?[branch] as? [String: Any],
              let ids = t["input_ids"] as? [Int], let segments = t["segments"] as? [[Int]] else {
            throw BreezeTTSError.badInput("\(name): no tokens for branch \(branch)")
        }
        return (ids.map(Int32.init), segments.map { $0.map(Int32.init) })
    }

    private func entry(_ name: String, precision: String?) -> [String: Any]? {
        let tensors: [String: Any]?
        if let precision {
            tensors = ((manifest["runs"] as? [String: Any])?[precision] as? [String: Any])?["tensors"] as? [String: Any]
        } else {
            tensors = (manifest["encoder"] as? [String: Any])?["tensors"] as? [String: Any]
        }
        return tensors?[name] as? [String: Any]
    }

    private func load(_ e: [String: Any]) throws -> (Data, [Int]) {
        let data = try Data(contentsOf: directory.appendingPathComponent(e["file"] as! String))
        let shape = e["shape"] as! [Int]
        guard data.count == shape.reduce(1, *) * 4 else { throw BreezeTTSError.badInput("\(name): \(e["file"]!) has \(data.count) bytes for shape \(shape)") }
        return (data, shape)
    }

    /// A float tensor of the fp32 or bf16 run, nil when the run has no such name.
    func tensor(_ name: String, precision: String = "fp32") throws -> Tensor? {
        guard let e = entry(name, precision: precision) else { return nil }
        let (data, shape) = try load(e)
        return Tensor(values: data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }, shape: shape)
    }

    func require(_ name: String, precision: String = "fp32") throws -> Tensor {
        guard let t = try tensor(name, precision: precision) else { throw BreezeTTSError.badInput("\(self.name): no tensor \(name)") }
        return t
    }

    func ints(_ name: String, precision: String? = "fp32") throws -> (values: [Int32], shape: [Int]) {
        guard let e = entry(name, precision: precision) else { throw BreezeTTSError.badInput("\(self.name): no tensor \(name)") }
        let (data, shape) = try load(e)
        return (data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }, shape)
    }

    /// The reference clip's codes as `prepare_inputs` put them in the prompt, nil without a clip.
    func referenceCodes() throws -> [[Int16]]? {
        guard entry("codes", precision: nil) != nil else { return nil }
        let (values, shape) = try ints("codes", precision: nil)
        return (0..<shape[0]).map { f in (0..<shape[1]).map { Int16(values[f * shape[1] + $0]) } }
    }

    /// The prompt exactly as the reference fed it: each branch's text segments and the clip's
    /// codes, placed where its `input_ids` hold the audio tokens.
    func referencePrompt() throws -> BreezePrompt {
        let clip = try referenceCodes()
        func branch(_ name: String) throws -> BreezePrompt.Branch {
            let (ids, texts) = try tokens(name)
            var segments: [BreezePrompt.Segment] = []
            var next = texts.makeIterator()
            var i = 0
            while i < ids.count {
                if ids[i] == BreezePrompt.audioTokenID {
                    guard let clip else { throw BreezeTTSError.badInput("\(self.name): audio tokens without a clip") }
                    guard ids.count >= i + clip.count + 1, ids[i + clip.count] == BreezePrompt.audioEndTokenID else {
                        throw BreezeTTSError.badInput("\(self.name): audio run does not match the clip's \(clip.count) frames")
                    }
                    segments.append(.audio(clip))
                    i += clip.count + 1
                } else {
                    guard let t = next.next(), Array(ids[i..<min(ids.count, i + t.count)]) == t else {
                        throw BreezeTTSError.badInput("\(self.name): text segments do not tile input_ids at \(i)")
                    }
                    segments.append(.text(t))
                    i += t.count
                }
            }
            return BreezePrompt.Branch(segments: segments)
        }
        let negative = guidance == 1 ? nil : BreezePrompt.Guidance(scale: guidance, negative: try branch("uncond"))
        return BreezePrompt(positive: try branch("cond"), guidance: negative)
    }

    /// The request the case was made from, for checking the front end.
    func request() throws -> BreezeRequest {
        let voice = try referenceCodes().map { try BreezeReferenceVoice(transcript: transcript ?? "", codes: $0) }
        let task: BreezeTask
        switch (voice, instruction) {
        case (nil, nil): task = .plain
        case (nil, let i?): task = .voiceDesign(instruction: i)
        case (let v?, nil): task = .voiceClone(v)
        case (let v?, let i?): task = .voiceDirection(v, instruction: i)
        }
        return try BreezeRequest(text: text ?? "", task: task)
    }
}
