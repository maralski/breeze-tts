// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Metal
import XCTest
@testable import BreezeTTS

/// The reference tensors of tools/reference (make_goldens.py and make_codec_extras.py), found
/// through BREEZE_GOLDENS. Tests that need them skip when it is unset.
enum CodecGoldens {
    static var root: URL? { ProcessInfo.processInfo.environment["BREEZE_GOLDENS"].map { URL(fileURLWithPath: $0) } }

    static func require() throws -> (goldens: URL, weights: URL) {
        guard let root else { throw XCTSkip("set BREEZE_GOLDENS to the reference goldens directory") }
        guard let weights = TestEnvironment.weightsDirectory else { throw XCTSkip("set BREEZE_WEIGHTS to the Breeze-TTS-2 snapshot") }
        return (root, weights)
    }

    static func floats(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    static func ints(_ url: URL) throws -> [Int32] {
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }

    static func codes(_ url: URL) throws -> [[Int]] {
        let flat = try ints(url)
        return stride(from: 0, to: flat.count, by: 16).map { i in (0..<16).map { Int(flat[i + $0]) } }
    }

    /// A golden stored [channels][time] as time-major rows.
    static func timeMajor(_ g: [Float], channels: Int) -> [Float] {
        let t = g.count / channels
        var out = [Float](repeating: 0, count: g.count)
        for c in 0..<channels {
            for i in 0..<t { out[i * channels + c] = g[c * t + i] }
        }
        return out
    }

    struct Agreement {
        let cosine: Double
        let maxAbs: Float
        let peak: Float
    }

    static func compare(_ got: [Float], _ want: [Float]) -> Agreement {
        precondition(got.count == want.count, "\(got.count) values against \(want.count)")
        var dot = 0.0, gg = 0.0, ww = 0.0
        var worst: Float = 0, peak: Float = 0
        for i in got.indices {
            dot += Double(got[i]) * Double(want[i])
            gg += Double(got[i]) * Double(got[i])
            ww += Double(want[i]) * Double(want[i])
            let d = abs(got[i] - want[i])
            worst = d.isNaN ? .infinity : max(worst, d)
            peak = max(peak, abs(want[i]))
        }
        return Agreement(cosine: dot / (gg * ww).squareRoot(), maxAbs: worst, peak: peak)
    }
}

final class CodecDecoderParityTests: XCTestCase {
    static let cases = ["design_en", "plain_en", "design_zh", "clone_en", "direction_en", "codec_extra/long"]
    /// Stage name, channels, and whether the golden is stored channel-major.
    static let stages: [(name: String, channels: Int, channelMajor: Bool)] = [
        ("quantizer", 512, true), ("pre_conv", 1024, true), ("pre_transformer", 1024, false), ("upsample0", 1024, true), ("upsample1", 1024, true),
        ("decoder0", 1536, true), ("decoder1", 768, true), ("decoder2", 384, true), ("decoder3", 192, true), ("decoder4", 96, true), ("decoder5", 96, true),
    ]

    nonisolated(unsafe) static var shared: (ctx: MetalContext, kernels: Kernels, codec: CodecKernels, weights: CodecDecoderWeights)?

    func environment() throws -> (ctx: MetalContext, kernels: Kernels, codec: CodecKernels, weights: CodecDecoderWeights, goldens: URL) {
        let env = try CodecGoldens.require()
        if Self.shared == nil {
            let ctx = GPUTestCase.ctx
            Self.shared = (ctx, GPUTestCase.kernels, try CodecKernels(ctx: ctx), try CodecDecoderWeights(try CodecLoader(directory: env.weights, ctx: ctx)))
        }
        let s = Self.shared!
        return (s.ctx, s.kernels, s.codec, s.weights, env.goldens)
    }

    func directory(_ goldens: URL, _ name: String) -> URL {
        name.contains("/") ? goldens.appendingPathComponent(name) : goldens.appendingPathComponent(name).appendingPathComponent("codec")
    }

    /// One frame per command buffer, every stage of every frame against one full forward of the
    /// reference. The long case runs 150 frames, past the 72-frame attention window.
    func testStreamingOneFrameAtATimeMatchesFullForward() throws {
        let env = try environment()
        let decoder = try CodecDecoder(ctx: env.ctx, kernels: env.codec, weights: env.weights)
        let codeBuffer = try env.ctx.makeBuffer(bytes: 64, label: "codes")
        for name in Self.cases {
            let dir = directory(env.goldens, name)
            let codes = try CodecGoldens.codes(dir.appendingPathComponent("codes.i32"))
            var got: [String: [Float]] = [:]
            var pcm: [Float] = []
            decoder.reset()
            for frame in codes {
                let p = codeBuffer.contents().bindMemory(to: UInt32.self, capacity: 16)
                for q in 0..<16 { p[q] = UInt32(frame[q]) }
                let step = decoder.append(codes: codeBuffer, codesOffset: 0)
                let cb = env.ctx.queue.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
                let e = Encoder(k: env.kernels, ce: ce)
                for d in step.dispatches { d(e) }
                ce.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()
                XCTAssertNil(cb.error)
                XCTAssertEqual(step.samples, 1920)
                for (stage, view) in decoder.lastStages { got[stage, default: []] += view.floats() }
                pcm += decoder.pcm.floats(count: step.samples, offset: step.pcmOffset)
            }
            for stage in Self.stages {
                let raw = try CodecGoldens.floats(dir.appendingPathComponent("\(stage.name).f32"))
                let want = stage.channelMajor ? CodecGoldens.timeMajor(raw, channels: stage.channels) : raw
                let a = CodecGoldens.compare(got[stage.name]!, want)
                print(String(format: "codec parity %@ %@ cosine %.9f maxAbs %.3g peak %.3g", name, stage.name, a.cosine, a.maxAbs, a.peak))
                XCTAssertGreaterThanOrEqual(a.cosine, 0.9999, "\(name) \(stage.name)")
                XCTAssertLessThanOrEqual(a.maxAbs, 2e-4 * max(1, a.peak), "\(name) \(stage.name)")
            }
            let a = CodecGoldens.compare(pcm, try CodecGoldens.floats(dir.appendingPathComponent("pcm_full.f32")))
            print(String(format: "codec parity %@ pcm frames %d cosine %.9f maxAbs %.3g peak %.3g", name, codes.count, a.cosine, a.maxAbs, a.peak))
            XCTAssertGreaterThanOrEqual(a.cosine, 0.9999, "\(name) pcm")
            XCTAssertLessThanOrEqual(a.maxAbs, 1e-4, "\(name) pcm")
        }
    }

    /// The same stream in steps of several frames, with a short last step, equals the
    /// one-frame stream: step size changes nothing, at 150 frames too.
    func testBatchDecodeEqualsOneFrameStream() throws {
        let env = try environment()
        for name in ["design_zh", "codec_extra/long"] {
            let dir = directory(env.goldens, name)
            let codes = try CodecGoldens.codes(dir.appendingPathComponent("codes.i32"))
            let want = try CodecGoldens.floats(dir.appendingPathComponent("pcm_full.f32"))
            let single = try CodecDecoder(ctx: env.ctx, kernels: env.codec, weights: env.weights).decode(codes: codes)
            for chunk in [7, 16] {
                let batch = try CodecDecoder(ctx: env.ctx, kernels: env.codec, weights: env.weights, chunkFrames: chunk).decode(codes: codes)
                let a = CodecGoldens.compare(batch, want), b = CodecGoldens.compare(batch, single)
                print(String(format: "codec batch %@ chunk %d vs golden maxAbs %.3g, vs one-frame stream maxAbs %.3g", name, chunk, a.maxAbs, b.maxAbs))
                XCTAssertLessThanOrEqual(a.maxAbs, 1e-4, "\(name) chunk \(chunk) against the reference")
                XCTAssertLessThanOrEqual(b.maxAbs, 2e-6, "\(name) chunk \(chunk) against the one-frame stream")
            }
        }
    }

    /// Frames committed ahead of the ones being read, codes read from per-frame slots of one
    /// buffer as the token loop lays them out, and a second stream after a reset.
    func testPipelinedFramesAndReset() throws {
        let env = try environment()
        let dir = directory(env.goldens, "plain_en")
        let codes = try CodecGoldens.codes(dir.appendingPathComponent("codes.i32"))
        let want = try CodecGoldens.floats(dir.appendingPathComponent("pcm_full.f32"))
        let decoder = try CodecDecoder(ctx: env.ctx, kernels: env.codec, weights: env.weights, pcmSlots: 4)
        let codeBuffer = try env.ctx.makeBuffer(bytes: codes.count * 64, label: "codes")
        let p = codeBuffer.contents().bindMemory(to: UInt32.self, capacity: codes.count * 16)
        for (i, frame) in codes.enumerated() { for q in 0..<16 { p[i * 16 + q] = UInt32(frame[q]) } }
        for pass in 0..<2 {
            decoder.reset()
            var pcm: [Float] = []
            var pending: [(MTLCommandBuffer, Int)] = []
            func drain() {
                let (cb, offset) = pending.removeFirst()
                cb.waitUntilCompleted()
                pcm += decoder.pcm.floats(count: 1920, offset: offset)
            }
            for i in codes.indices {
                let step = decoder.append(codes: codeBuffer, codesOffset: i * 64)
                let cb = env.ctx.queue.makeCommandBuffer()!, ce = cb.makeComputeCommandEncoder()!
                let e = Encoder(k: env.kernels, ce: ce)
                for d in step.dispatches { d(e) }
                ce.endEncoding()
                cb.commit()
                pending.append((cb, step.pcmOffset))
                if pending.count == 3 { drain() }
            }
            while !pending.isEmpty { drain() }
            XCTAssertLessThanOrEqual(CodecGoldens.compare(pcm, want).maxAbs, 1e-4, "pass \(pass)")
        }
    }
}

final class CodecEncoderParityTests: XCTestCase {
    /// The reference clip (a whole number of frames), a clip cut mid-frame with an odd 25 Hz
    /// length, and an 8.3 s clip: every stage against the reference, codes exactly.
    func testEncoderMatchesReferenceCodes() throws {
        let env = try CodecGoldens.require()
        let ctx = GPUTestCase.ctx
        let encoder = try CodecEncoder(ctx: ctx, kernels: try CodecKernels(ctx: ctx), weights: try CodecEncoderWeights(try CodecLoader(directory: env.weights, ctx: ctx)), engine: GPUTestCase.kernels)
        for name in ["clone_en/encoder", "direction_en/encoder", "codec_extra/enc_odd", "codec_extra/enc_long"] {
            let dir = env.goldens.appendingPathComponent(name)
            let samples = try CodecGoldens.floats(dir.appendingPathComponent("input_samples.f32"))
            let want = try CodecGoldens.codes(dir.appendingPathComponent("codes.i32"))
            let t0 = Date()
            let got = try encoder.encodeStages(samples: samples)
            let seconds = Date().timeIntervalSince(t0)
            XCTAssertEqual(got.codes.count, (samples.count + 1919) / 1920, name)
            let goldenDown = CodecGoldens.timeMajor(try CodecGoldens.floats(dir.appendingPathComponent("downsample.f32")), channels: 512)
            for (stage, values, golden) in [("seanet", got.seanet, CodecGoldens.timeMajor(try CodecGoldens.floats(dir.appendingPathComponent("seanet.f32")), channels: 512)),
                                            ("transformer", got.transformer, try CodecGoldens.floats(dir.appendingPathComponent("transformer.f32"))),
                                            ("downsample", got.downsample, goldenDown)] {
                let a = CodecGoldens.compare(values, golden)
                print(String(format: "encoder parity %@ %@ cosine %.9f maxAbs %.3g peak %.3g", name, stage, a.cosine, a.maxAbs, a.peak))
                XCTAssertGreaterThanOrEqual(a.cosine, 0.9999, "\(name) \(stage)")
                XCTAssertLessThanOrEqual(a.maxAbs, 2e-4 * a.peak, "\(name) \(stage)")
            }
            let fromGolden = try encoder.quantize(embedding: goldenDown)
            let flatWant = want.flatMap { $0 }
            let endToEnd = zip(got.codes.flatMap { $0 }, flatWant).filter { Int($0) != $1 }.count
            let quantizerOnly = zip(fromGolden.flatMap { $0 }, flatWant).filter { Int($0) != $1 }.count
            print(String(format: "encoder parity %@ %d samples, %d frames in %.3f s: %d of %d codes differ end to end, %d from the reference embedding",
                         name, samples.count, want.count, seconds, endToEnd, flatWant.count, quantizerOnly))
            XCTAssertEqual(got.codes.map { $0.map(Int.init) }, want, "\(name) end to end")
            XCTAssertEqual(fromGolden.map { $0.map(Int.init) }, want, "\(name) quantizer on the reference embedding")
        }
    }
}
