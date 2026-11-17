// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import BreezeTTS

final class WeightsTests: GPUTestCase {
    /// Writes a safetensors file holding `tensors` in order.
    static func writeSafeTensors(_ url: URL, _ tensors: [(name: String, dtype: String, shape: [Int], bytes: Data)]) throws {
        var header: [String: Any] = ["__metadata__": ["format": "pt"]]
        var offset = 0
        for t in tensors {
            header[t.name] = ["dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.bytes.count]]
            offset += t.bytes.count
        }
        var json = try JSONSerialization.data(withJSONObject: header)
        json.append(Data(repeating: 0x20, count: (8 - json.count % 8) % 8))
        var file = withUnsafeBytes(of: UInt64(json.count).littleEndian) { Data($0) }
        file.append(json)
        for t in tensors { file.append(t.bytes) }
        try file.write(to: url)
    }

    static func bf16Data(_ values: [Float]) -> Data {
        values.map(bf16Bits).withUnsafeBytes { Data($0) }
    }

    func makeCheckpoint(_ shards: [String: [(name: String, dtype: String, shape: [Int], bytes: Data)]], map: [String: String]? = nil) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("breeze-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        var weightMap: [String: String] = [:]
        for (file, tensors) in shards {
            try Self.writeSafeTensors(dir.appendingPathComponent(file), tensors)
            for t in tensors { weightMap[t.name] = file }
        }
        let index: [String: Any] = ["metadata": ["total_size": 0], "weight_map": map ?? weightMap]
        try JSONSerialization.data(withJSONObject: index).write(to: dir.appendingPathComponent(ShardedCheckpoint.indexName))
        return dir
    }

    /// Tensors come back by name from whichever shard the index names, at any byte alignment.
    func testShardedCheckpoint() throws {
        var rng = RNG(state: 5)
        let a = rng.floats(6), b = rng.floats(12), c = rng.floats(4)
        let dir = try makeCheckpoint([
            // A one-element tensor first leaves `a.weight` at an offset that is 2 mod 16.
            "model-00001-of-00002.safetensors": [("odd", "BF16", [1], Self.bf16Data([1])), ("a.weight", "BF16", [2, 3], Self.bf16Data(a))],
            "model-00002-of-00002.safetensors": [("b.weight", "BF16", [3, 4], Self.bf16Data(b)), ("c", "F32", [4], c.withUnsafeBytes { Data($0) })],
        ])
        let ckpt = try ShardedCheckpoint(directory: dir)
        XCTAssertEqual(try ckpt.floats("a.weight"), a.map(Self.bf16Round))
        XCTAssertEqual(try ckpt.floats("b.weight"), b.map(Self.bf16Round))
        XCTAssertEqual(try ckpt.floats("c"), c)
        XCTAssertEqual(try ckpt.entry("b.weight").shape, [3, 4])
        XCTAssertThrowsError(try ckpt.entry("missing"))
        let stacked = try WeightArena.load(ckpt, ctx: ctx, label: "t") { try $0.stackedLinear(["a", "a"], n: [2, 2], k: 3) }
        XCTAssertEqual(stacked.n, 4)
        let bits = UnsafeBufferPointer(start: (stacked.w.buffer.contents() + stacked.w.offset).bindMemory(to: UInt16.self, capacity: 12), count: 12)
        XCTAssertEqual(Array(bits), (a + a).map(Self.bf16Bits))
        XCTAssertThrowsError(try WeightArena.load(ckpt, ctx: ctx, label: "t") { try $0.linear("b", n: 4, k: 3) }, "a wrong shape is refused")
        XCTAssertThrowsError(try WeightArena.load(ckpt, ctx: ctx, label: "t") { try $0.tensor("c") }, "fp32 is refused where bf16 is expected")
    }

    func testIndexMustMatchShards() throws {
        let dir = try makeCheckpoint(["s1.safetensors": [("a", "BF16", [1], Self.bf16Data([1]))], "s2.safetensors": [("b", "BF16", [1], Self.bf16Data([1]))]],
                                     map: ["a": "s2.safetensors", "b": "s2.safetensors"])
        XCTAssertThrowsError(try ShardedCheckpoint(directory: dir))
        XCTAssertThrowsError(try ShardedCheckpoint(directory: dir.appendingPathComponent("nowhere")))
    }

    /// The codebook heads are stored [head, in, out]. The reference here indexes that stored
    /// layout directly; the engine runs its transposed copy through the ordinary gemv.
    func testTransposedHeads() throws {
        let count = 3, inW = 64, outW = 37
        var rng = RNG(state: 9)
        let stored = rng.floats(count * inW * outW, scale: 0.2)
        let dir = try makeCheckpoint(["m.safetensors": [("pad", "BF16", [1], Self.bf16Data([0])), ("heads", "BF16", [count, inW, outW], Self.bf16Data(stored))]])
        let heads = try WeightArena.load(try ShardedCheckpoint(directory: dir), ctx: ctx, label: "heads") {
            try $0.transposedHeads("heads", count: count, inWidth: inW, outWidth: outW)
        }
        let x = rng.floats(2 * inW)
        let xb = try f32Buffer(x + [Float](repeating: .nan, count: 8 * inW)), yb = try nanBuffer(floats: count * 2 * outW)
        try run { e in for i in 0..<count { e.linear(ActView(xb, ld: inW), rows: 2, heads[i], ActView(yb, ld: outW).row(2 * i)) } }
        var want = [Float](repeating: 0, count: count * 2 * outW)
        let w = stored.map(Self.bf16Round)
        for i in 0..<count {
            for r in 0..<2 {
                for n in 0..<outW {
                    for k in 0..<inW { want[(2 * i + r) * outW + n] += x[r * inW + k] * w[(i * inW + k) * outW + n] }
                }
            }
        }
        assertClose(yb.floats(count: want.count), want, atol: 1e-5, "codebook heads")
    }

    /// The three RoPE tables against values computed independently (the spec's numpy figures
    /// for llama3, and the defining formulas).
    func testRopeTables() {
        let depth = DecoderConfig.depth.rope.values
        XCTAssertEqual(depth.count, 64)
        XCTAssertEqual(depth[0], 1)
        XCTAssertEqual(depth[28], 1 / powf(5e5, 56.0 / 128), accuracy: 1e-9)
        XCTAssertEqual(depth[29], 0.0021184, accuracy: 2e-7)
        XCTAssertEqual(depth[32], 0.00042956, accuracy: 5e-8)
        XCTAssertEqual(depth[34], 9.7083e-05, accuracy: 1e-8)
        XCTAssertEqual(depth[63], 1 / powf(5e5, 126.0 / 128) / 32, accuracy: 1e-12)
        let backbone = DecoderConfig.backbone.rope.values
        XCTAssertEqual(backbone.count, 64)
        XCTAssertEqual(backbone[63], 1 / powf(1e6, 126.0 / 128), accuracy: 1e-12)
        let full = RopeFrequencies.linear(theta: 1e6, dim: 256, factor: 8).values
        XCTAssertEqual(full.count, 128)
        XCTAssertEqual(full[0], 0.125)
        XCTAssertEqual(full[127], 1 / (8 * powf(1e6, 254.0 / 256)), accuracy: 1e-12)
    }

    /// The facts the engine relies on, read from the real checkpoint's headers. Needs
    /// BREEZE_WEIGHTS or the Hugging Face cache.
    func testRealCheckpointHeaders() throws {
        guard let dir = TestEnvironment.weightsDirectory else { throw XCTSkip("set BREEZE_WEIGHTS to the Breeze-TTS-2 snapshot") }
        let ckpt = try ShardedCheckpoint(directory: dir)
        XCTAssertEqual(ckpt.names.count, 1115)
        func shape(_ n: String) throws -> [Int] { try ckpt.entry(n).shape }
        XCTAssertEqual(try shape("depth_decoder.codebooks_head.weight"), [15, 1024, 2051])
        XCTAssertEqual(try shape("depth_decoder.model.embed_tokens.weight"), [32816, 2048])
        XCTAssertEqual(try shape("lm_head.weight"), [2052, 2048])
        XCTAssertEqual(try shape("depth_decoder.model.inputs_embeds_projector.weight"), [1024, 2048])
        XCTAssertTrue(ckpt.names.filter { $0.hasPrefix("backbone_model.") && $0.contains("embed") }.isEmpty, "the backbone has no embedding copy")
        XCTAssertEqual(ckpt.names.filter { $0.hasPrefix("backbone_model.") && ($0.contains("q_norm") || $0.contains("k_norm")) }.count, 56)
        XCTAssertEqual(ckpt.names.filter { $0.hasPrefix("depth_decoder.") && ($0.contains("q_norm") || $0.contains("k_norm")) }.count, 0)
        XCTAssertEqual(try shape("backbone_model.layers.27.self_attn.q_norm.weight"), [128])
        XCTAssertTrue(ckpt.names.filter { $0.hasSuffix(".bias") && !$0.hasPrefix("codec_model.") }.isEmpty, "no biases outside the unused codec")
        XCTAssertTrue(try ckpt.names.filter { !$0.hasPrefix("codec_model.") }.allSatisfy { try ckpt.entry($0).dtype == "BF16" })
        let codec = try SafeTensors(url: dir.appendingPathComponent("audio_tokenizer/model.safetensors"))
        XCTAssertEqual(codec.entries.count, 496)
        XCTAssertTrue(codec.entries.values.allSatisfy { $0.dtype == "F32" })
    }
}

enum TestEnvironment {
    /// BREEZE_WEIGHTS, else the newest complete Breeze-TTS-2 snapshot in the Hugging Face cache.
    static let weightsDirectory: URL? = BreezeCheckpoint.find()?.directory
}
