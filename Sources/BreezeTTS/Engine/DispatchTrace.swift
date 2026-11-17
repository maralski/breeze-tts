// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import Foundation
import Metal

/// GPU time per dispatch, for attribution only.
///
/// Apple GPUs sample counters at pass boundaries, so a traced `Encoder` gives every dispatch a
/// compute pass of its own with a timestamp at each end. A pass boundary costs time a real frame
/// does not pay, so the trace says where a frame's time goes, not how long the frame is: its
/// rows are scaled so they sum to the span the command buffer reports.
final class DispatchTrace {
    struct Row {
        let kernel: String
        let weightBytes: Int
        var count = 0
        var seconds = 0.0
    }

    private let commandBuffer: MTLCommandBuffer
    /// Metal caps a sample buffer at 32 KB: 2048 passes each.
    private let samples: [MTLCounterSampleBuffer]
    private static let passesPerBuffer = 2048
    private let names: [ObjectIdentifier: String]
    private let capacity: Int
    private var passes: [(kernel: String, weightBytes: Int)] = []

    /// nil where the device cannot timestamp pass boundaries.
    init?(ctx: MetalContext, commandBuffer: MTLCommandBuffer, capacity: Int) throws {
        guard ctx.device.supportsCounterSampling(.atStageBoundary),
              let set = ctx.device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else { return nil }
        let d = MTLCounterSampleBufferDescriptor()
        d.counterSet = set
        d.storageMode = .shared
        d.sampleCount = 2 * Self.passesPerBuffer
        samples = try (0..<((capacity + Self.passesPerBuffer - 1) / Self.passesPerBuffer)).map { _ in try ctx.device.makeCounterSampleBuffer(descriptor: d) }
        self.commandBuffer = commandBuffer
        self.capacity = capacity
        var names: [ObjectIdentifier: String] = [:]
        for name in ctx.library.functionNames { names[ObjectIdentifier(try ctx.pipeline(name))] = name }
        self.names = names
    }

    /// A pass that is never sampled: what a traced `Encoder` holds before its first dispatch.
    func idlePass() -> MTLComputeCommandEncoder { commandBuffer.makeComputeCommandEncoder()! }

    /// The pass of the next dispatch.
    func pass(for pipeline: MTLComputePipelineState, weightBytes: Int) -> MTLComputeCommandEncoder {
        precondition(passes.count < capacity, "the trace holds \(capacity) dispatches")
        let d = MTLComputePassDescriptor()
        let a = d.sampleBufferAttachments[0]!
        let slot = passes.count % Self.passesPerBuffer
        a.sampleBuffer = samples[passes.count / Self.passesPerBuffer]
        a.startOfEncoderSampleIndex = 2 * slot
        a.endOfEncoderSampleIndex = 2 * slot + 1
        passes.append((names[ObjectIdentifier(pipeline)] ?? "?", weightBytes))
        return commandBuffer.makeComputeCommandEncoder(descriptor: d)!
    }

    /// Call after the command buffer has completed. Rows by (kernel, weight bytes), largest first.
    func rows() throws -> [Row] {
        var stamps: [UInt64] = []
        for (i, buffer) in samples.enumerated() {
            let used = min(passes.count - i * Self.passesPerBuffer, Self.passesPerBuffer)
            guard used > 0, let data = try buffer.resolveCounterRange(0..<(2 * used)) else { break }
            stamps += data.withUnsafeBytes { Array($0.bindMemory(to: MTLCounterResultTimestamp.self)) }.map(\.timestamp)
        }
        guard stamps.count == 2 * passes.count else { return [] }
        var ticks = 0.0
        var byKey: [String: Row] = [:]
        for (i, p) in passes.enumerated() {
            let t = Double(stamps[2 * i + 1] &- stamps[2 * i])
            ticks += t
            byKey["\(p.kernel) \(p.weightBytes)", default: Row(kernel: p.kernel, weightBytes: p.weightBytes)].count += 1
            byKey["\(p.kernel) \(p.weightBytes)"]!.seconds += t
        }
        let scale = ticks > 0 ? (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) / ticks : 0
        return byKey.values.map { var r = $0; r.seconds *= scale; return r }.sorted { $0.seconds > $1.seconds }
    }

    static func report(_ rows: [Row], frames: Int) -> String {
        var lines: [String] = []
        let total = rows.reduce(0) { $0 + $1.seconds }
        for r in rows {
            let us = r.seconds / Double(r.count) * 1e6
            let mb = Double(r.weightBytes) / 1e6
            lines.append(String(format: "  %@ %7.2f MB x%-4d %7.3f ms per frame %5.1f%%  %6.1f us each%@",
                                r.kernel.padding(toLength: 22, withPad: " ", startingAt: 0), mb, r.count / frames, r.seconds / Double(frames) * 1e3,
                                r.seconds / total * 100, us, r.weightBytes > 1 << 20 ? String(format: "  %4.0f GB/s", mb / us * 1e3) : ""))
        }
        return lines.joined(separator: "\n")
    }
}
