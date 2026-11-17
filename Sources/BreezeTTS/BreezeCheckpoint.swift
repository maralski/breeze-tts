// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// A folder that should hold the Breeze TTS 2 checkpoint. Check it before loading, so a
/// partial download is reported by the names of the files it lacks.
public struct BreezeCheckpoint: Sendable, Equatable {
    /// The Hugging Face repository the checkpoint is published in.
    public static let repository = "BreezeBlue/Breeze-TTS-2"
    /// The files `BreezeTTSModel` reads to speak: the sharded backbone, text encoder and depth
    /// decoder, the tokenizer, and the audio tokenizer whose decoder turns codes into audio.
    public static let synthesisFiles = [
        "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "tokenizer.json", "audio_tokenizer/model.safetensors",
    ]
    /// The file `BreezeTTSModel.encodeReference` reads to clone a voice: the audio tokenizer,
    /// whose encoder turns a clip into codes. It is a synthesis file too.
    public static let cloningFiles = ["audio_tokenizer/model.safetensors"]

    public let directory: URL
    public init(directory: URL) { self.directory = directory }

    /// The files of `synthesisFiles` and `cloningFiles` the folder lacks, in that order, each once.
    public var missingFiles: [String] {
        var seen = Set<String>()
        return missing((Self.synthesisFiles + Self.cloningFiles).filter { seen.insert($0).inserted })
    }

    public var isComplete: Bool { missingFiles.isEmpty }

    func missing(_ files: [String]) -> [String] {
        files.filter { !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    /// - Throws: `BreezeTTSError.badCheckpoint` naming the folder and each of `files` it lacks.
    func require(_ files: [String]) throws {
        let missing = missing(files)
        guard missing.isEmpty else {
            throw BreezeTTSError.badCheckpoint("\(directory.path) lacks \(missing.joined(separator: ", "))")
        }
    }

    /// The folder `BREEZE_WEIGHTS` names, complete or not, so a caller can say what it lacks.
    /// Else the newest complete snapshot of `repository` in the Hugging Face cache, or the newest
    /// one when none is complete. Nil when there is neither.
    public static func find(environment: [String: String] = ProcessInfo.processInfo.environment) -> BreezeCheckpoint? {
        if let path = environment["BREEZE_WEIGHTS"], !path.isEmpty {
            return BreezeCheckpoint(directory: URL(fileURLWithPath: path, isDirectory: true))
        }
        let snapshots = huggingFaceCache(environment)
            .appendingPathComponent("models--" + repository.replacingOccurrences(of: "/", with: "--"))
            .appendingPathComponent("snapshots")
        let keys: Set<URLResourceKey> = [.contentModificationDateKey, .isDirectoryKey]
        let found = (try? FileManager.default.contentsOfDirectory(at: snapshots, includingPropertiesForKeys: Array(keys))) ?? []
        let newestFirst = found
            .filter { (try? $0.resourceValues(forKeys: keys).isDirectory) == true }
            .sorted { modified($0) > modified($1) }
            .map(BreezeCheckpoint.init(directory:))
        return newestFirst.first(where: \.isComplete) ?? newestFirst.first
    }

    /// `HF_HUB_CACHE`, else `hub` in `HF_HOME`, else `~/.cache/huggingface/hub`, as `huggingface_hub` resolves it.
    static func huggingFaceCache(_ environment: [String: String]) -> URL {
        if let cache = environment["HF_HUB_CACHE"], !cache.isEmpty { return URL(fileURLWithPath: cache, isDirectory: true) }
        let home = environment["HF_HOME"].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface")
        return home.appendingPathComponent("hub")
    }

    private static func modified(_ url: URL) -> Date {
        (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
    }
}
