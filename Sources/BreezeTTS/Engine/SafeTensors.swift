// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation

/// One tensor of a checkpoint: where its bytes are and what they mean.
struct TensorEntry {
    let dtype: String
    let shape: [Int]
    let byteRange: Range<Int>
    var count: Int { shape.reduce(1, *) }
}

/// Tensors by name, whatever file each lives in.
protocol TensorStore {
    func entry(_ name: String) throws -> TensorEntry
    /// Calls `body` with the tensor's raw bytes.
    func withBytes<R>(_ name: String, _ body: (UnsafeRawBufferPointer, TensorEntry) throws -> R) throws -> R
}

extension TensorStore {
    /// Decodes a bf16 or f32 tensor to host floats.
    func floats(_ name: String) throws -> [Float] {
        try withBytes(name) { raw, e in
            switch e.dtype {
            case "BF16": return raw.bindMemory(to: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            case "F32": return Array(raw.bindMemory(to: Float.self))
            default: throw BreezeTTSError.badCheckpoint("\(name) has dtype \(e.dtype)")
            }
        }
    }
}

/// A memory-mapped safetensors file.
final class SafeTensors: TensorStore {
    let entries: [String: TensorEntry]
    private let data: Data
    private let dataStart: Int

    init(url: URL) throws {
        do {
            data = try Data(contentsOf: url, options: .alwaysMapped)
        } catch {
            throw BreezeTTSError.badCheckpoint("cannot map \(url.lastPathComponent): \(error.localizedDescription)")
        }
        guard data.count >= 8 else { throw BreezeTTSError.badCheckpoint("\(url.lastPathComponent) is too small") }
        let headerLength = Int(data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard headerLength > 0, 8 + headerLength <= data.count else {
            throw BreezeTTSError.badCheckpoint("\(url.lastPathComponent): header length \(headerLength) exceeds file")
        }
        guard let json = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + headerLength))) as? [String: Any] else {
            throw BreezeTTSError.badCheckpoint("\(url.lastPathComponent): header is not a JSON object")
        }
        dataStart = 8 + headerLength
        var entries: [String: TensorEntry] = [:]
        for (name, value) in json where name != "__metadata__" {
            guard let dict = value as? [String: Any],
                  let dtype = dict["dtype"] as? String,
                  let shape = dict["shape"] as? [Int],
                  let offsets = dict["data_offsets"] as? [Int], offsets.count == 2, offsets[0] <= offsets[1] else {
                throw BreezeTTSError.badCheckpoint("malformed entry \(name)")
            }
            entries[name] = TensorEntry(dtype: dtype, shape: shape, byteRange: offsets[0]..<offsets[1])
        }
        self.entries = entries
        if let last = entries.values.map(\.byteRange.upperBound).max(), dataStart + last > data.count {
            throw BreezeTTSError.badCheckpoint("\(url.lastPathComponent) holds \(data.count) bytes, header needs \(dataStart + last); download incomplete?")
        }
    }

    func entry(_ name: String) throws -> TensorEntry {
        guard let e = entries[name] else { throw BreezeTTSError.missingTensor(name) }
        return e
    }

    func withBytes<R>(_ name: String, _ body: (UnsafeRawBufferPointer, TensorEntry) throws -> R) throws -> R {
        let e = try entry(name)
        return try data.withUnsafeBytes { raw in
            try body(UnsafeRawBufferPointer(rebasing: raw[(dataStart + e.byteRange.lowerBound)..<(dataStart + e.byteRange.upperBound)]), e)
        }
    }
}

/// A checkpoint split over several safetensors shards, joined by `model.safetensors.index.json`,
/// whose `weight_map` names the shard of every tensor. Shards are mapped, never read whole.
final class ShardedCheckpoint: TensorStore {
    static let indexName = "model.safetensors.index.json"
    /// Tensor name to shard file name, as the index states it.
    let weightMap: [String: String]
    private let shards: [String: SafeTensors]

    init(directory: URL) throws {
        let indexURL = directory.appendingPathComponent(Self.indexName)
        guard let raw = try? Data(contentsOf: indexURL),
              let json = try? JSONSerialization.jsonObject(with: raw) as? [String: Any],
              let map = json["weight_map"] as? [String: String] else {
            throw BreezeTTSError.badCheckpoint("\(indexURL.path) is missing or has no weight_map")
        }
        weightMap = map
        var shards: [String: SafeTensors] = [:]
        for file in Set(map.values) { shards[file] = try SafeTensors(url: directory.appendingPathComponent(file)) }
        self.shards = shards
        for (name, file) in map where shards[file]!.entries[name] == nil {
            throw BreezeTTSError.badCheckpoint("index places \(name) in \(file), which does not hold it")
        }
    }

    var names: [String] { Array(weightMap.keys) }

    private func shard(_ name: String) throws -> SafeTensors {
        guard let file = weightMap[name], let s = shards[file] else { throw BreezeTTSError.missingTensor(name) }
        return s
    }

    func entry(_ name: String) throws -> TensorEntry { try shard(name).entry(name) }

    func withBytes<R>(_ name: String, _ body: (UnsafeRawBufferPointer, TensorEntry) throws -> R) throws -> R {
        try shard(name).withBytes(name, body)
    }
}
