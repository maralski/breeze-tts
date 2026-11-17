// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
// Usage:
//   breeze-tts --bench-frame [--weights DIR] [--rounds N] [--frames N] [--concurrent] [--guidance S]
//   breeze-tts --bench-gemv [--repeats N] [--rounds N]
//   breeze-tts --parity CASE_DIR [--weights DIR]   (a case of the reference goldens)
// --weights defaults to BREEZE_WEIGHTS, then the Hugging Face cache. Run benchmarks under
// scripts/gpulock.py so they never overlap.
import BreezeTTS
import Foundation

enum Mode { case benchFrame, benchGemv, parity(URL) }

do {
    var mode: Mode?
    var weights: URL?
    var frame = FrameBench.Options()
    var repeats = 100, rounds: Int?
    var args = CommandLine.arguments.dropFirst().makeIterator()
    func value(_ flag: String) throws -> String {
        guard let v = args.next() else { throw BreezeTTSError.badInput("\(flag) needs a value") }
        return v
    }
    func int(_ flag: String) throws -> Int {
        guard let v = Int(try value(flag)), v > 0 else { throw BreezeTTSError.badInput("\(flag) needs a positive integer") }
        return v
    }
    while let arg = args.next() {
        switch arg {
        case "--bench-frame": mode = .benchFrame
        case "--bench-gemv": mode = .benchGemv
        case "--parity": mode = .parity(URL(fileURLWithPath: try value(arg)))
        case "--weights": weights = URL(fileURLWithPath: try value(arg))
        case "--rounds": rounds = try int(arg)
        case "--frames": frame.frames = max(2, try int(arg))
        case "--repeats": repeats = try int(arg)
        case "--concurrent": frame.concurrent = true
        case "--guidance":
            guard let s = Float(try value(arg)) else { throw BreezeTTSError.badInput("--guidance needs a number") }
            frame.guidance = s
        default: throw BreezeTTSError.badInput("unknown argument \(arg)")
        }
    }
    switch mode {
    case .benchFrame:
        guard let dir = weights ?? FrameBench.defaultWeights() else {
            throw BreezeTTSError.badCheckpoint("no checkpoint: pass --weights or set BREEZE_WEIGHTS")
        }
        if let rounds { frame.rounds = rounds }
        try FrameBench.run(directory: dir, options: frame)
    case .benchGemv:
        try GemvBench.run(repeats: repeats, rounds: rounds ?? 5)
    case .parity(let caseDirectory):
        guard let dir = weights ?? FrameBench.defaultWeights() else {
            throw BreezeTTSError.badCheckpoint("no checkpoint: pass --weights or set BREEZE_WEIGHTS")
        }
        let report = try BreezeParity(weights: dir).run(caseDirectory: caseDirectory)
        print(report.description)
        if !report.passed { exit(1) }
    case nil:
        print("usage: breeze-tts --bench-frame [--weights DIR] [--rounds N] [--frames N] [--concurrent] | --bench-gemv [--repeats N] [--rounds N] | --parity CASE_DIR [--weights DIR]")
    }
} catch {
    FileHandle.standardError.write("breeze-tts: \(error)\n".data(using: .utf8)!)
    exit(2)
}
