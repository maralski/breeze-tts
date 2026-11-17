// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import BreezeTTS
import Foundation

let usage = """
    breeze-tts: Breeze TTS 2 on the GPU.

      breeze-tts --say TEXT [--out FILE.wav] [voice] [sampling]
          Speaks TEXT into a 24 kHz mono float WAV (default out.wav). When the text splits into
          fragments, each fragment is also written as FILE.fragN.wav.
        voice (default: the model's own voice):
          --instruction TEXT             voice design from a description
          --clone REF.wav --transcript TEXT
                                         clone the voice in REF.wav (silence is trimmed first)
          --clone ... --instruction TEXT voice direction: the cloned voice, directed
        sampling:
          --guidance S                   guidance scale (default 4 for design and direction, 1 otherwise;
                                         plain and clone accept only 1)
          --seed N                       seed of the draws (default 0)
          --greedy                       argmax instead of sampling (for testing)
          --max-frames N                 frames per fragment, 80 ms each (default 1500)
          --fragment-length N            words or CJK characters per fragment (default 60)

      breeze-tts --bench [--repeat N]
          End-to-end speed on two fixed sentences, plain (unguided) and voice design (guided),
          one warm-up run then N runs with seeds 1...N (default 5). Run it under scripts/gpulock.py.

      breeze-tts --parity CASE_DIR       stage-by-stage parity against one reference golden case
      breeze-tts --bench-frame | --bench-gemv | --bench-prompt | --bench-codec [--token-loop]
          kernel benchmarks, with [--rounds N] [--frames N] [--concurrent] [--guidance S] [--repeats N]

      --weights DIR                      the Breeze-TTS-2 snapshot; default BREEZE_WEIGHTS, then the
                                         Hugging Face cache
      --help                             this text

    Errors print to stderr with exit code 2. --parity exits 1 when the case fails.
    """

enum Mode { case say(String), bench, benchFrame, benchGemv, benchPrompt, parity(URL), benchCodec }

struct Args {
    var mode: Mode?
    var weights: URL?
    var out = "out.wav"
    var instruction: String?
    var clone: String?
    var transcript: String?
    var options = BreezeSynthesisOptions()
    var repeats = 5
    var frame = FrameBench.Options()
    var codec = CodecBench.Options()
    var gemvRepeats = 100, rounds: Int?, frames: Int?
    var help = false

    init(_ argv: ArraySlice<String>) throws {
        var it = argv.makeIterator()
        func value(_ flag: String) throws -> String {
            guard let v = it.next() else { throw BreezeTTSError.badInput("\(flag) needs a value") }
            return v
        }
        func int(_ flag: String) throws -> Int {
            guard let v = Int(try value(flag)), v > 0 else { throw BreezeTTSError.badInput("\(flag) needs a positive integer") }
            return v
        }
        while let arg = it.next() {
            switch arg {
            case "--say": mode = .say(try value(arg))
            case "--out": out = try value(arg)
            case "--instruction": instruction = try value(arg)
            case "--clone": clone = try value(arg)
            case "--transcript": transcript = try value(arg)
            case "--seed":
                guard let s = UInt64(try value(arg)) else { throw BreezeTTSError.badInput("--seed needs a non-negative integer") }
                options.seed = s
            case "--greedy": options.greedy = true
            case "--max-frames": options.maxFrames = try int(arg)
            case "--fragment-length": options.fragmentLength = try int(arg)
            case "--bench": mode = .bench
            case "--repeat": repeats = try int(arg)
            case "--parity": mode = .parity(URL(fileURLWithPath: try value(arg)))
            case "--bench-frame": mode = .benchFrame
            case "--bench-gemv": mode = .benchGemv
            case "--bench-prompt": mode = .benchPrompt
            case "--bench-codec": mode = .benchCodec
            case "--token-loop": codec.tokenLoop = true
            case "--weights": weights = URL(fileURLWithPath: try value(arg))
            case "--rounds": rounds = try int(arg)
            case "--frames":
                frames = max(2, try int(arg))
                frame.frames = frames!
            case "--repeats": gemvRepeats = try int(arg)
            case "--concurrent":
                frame.concurrent = true
                codec.concurrent = true
            case "--guidance":
                guard let s = Float(try value(arg)) else { throw BreezeTTSError.badInput("--guidance needs a number") }
                options.guidance = s
                frame.guidance = s
                codec.guidance = s
            case "--help", "-h": help = true
            default: throw BreezeTTSError.badInput("unknown argument \(arg); see --help")
            }
        }
    }

    func checkpoint() throws -> URL {
        guard let dir = weights ?? BreezeCheckpoint.find()?.directory else {
            throw BreezeTTSError.badCheckpoint("no checkpoint: pass --weights DIR or set BREEZE_WEIGHTS; none is in the Hugging Face cache")
        }
        return dir
    }

    func loadModel() throws -> BreezeTTSModel {
        let dir = try checkpoint()
        let t0 = Date()
        let model = try BreezeTTSModel(directory: dir)
        print("loaded \(dir.path) in \(fmt(Date().timeIntervalSince(t0), 2)) s")
        return model
    }

    /// The request the voice flags describe.
    func request(text: String, model: BreezeTTSModel) throws -> BreezeRequest {
        guard let clone else {
            if transcript != nil { throw BreezeTTSError.badInput("--transcript needs --clone") }
            return try BreezeRequest(text: text, task: instruction.map { .voiceDesign(instruction: $0) } ?? .plain)
        }
        guard let transcript else { throw BreezeTTSError.badInput("--clone needs --transcript") }
        let (samples, rate) = try BreezeTTSModel.readMono(URL(fileURLWithPath: clone))
        let t0 = Date()
        let voice = try model.encodeReference(samples: trimSilence(samples, sampleRate: rate), sampleRate: rate, transcript: transcript)
        print("encoded the reference clip (\(fmt(voice.seconds, 2)) s, \(voice.frames) frames) in \(fmt(Date().timeIntervalSince(t0) * 1000, 0)) ms")
        return try BreezeRequest(text: text, task: instruction.map { .voiceDirection(voice, instruction: $0) } ?? .voiceClone(voice))
    }
}

func fmt(_ x: Double, _ digits: Int = 3) -> String { String(format: "%.\(digits)f", x) }

func writeWav(_ samples: [Float], to path: String, sampleRate: Int = BreezeTTSModel.sampleRate) throws {
    var d = Data()
    func u32(_ v: UInt32) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 4)) }
    func u16(_ v: UInt16) { var x = v.littleEndian; d.append(Data(bytes: &x, count: 2)) }
    let bytes = samples.count * 4
    d.append("RIFF".data(using: .ascii)!); u32(UInt32(36 + bytes)); d.append("WAVE".data(using: .ascii)!)
    d.append("fmt ".data(using: .ascii)!); u32(16); u16(3); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 4)); u16(4); u16(32)
    d.append("data".data(using: .ascii)!); u32(UInt32(bytes))
    samples.withUnsafeBytes { d.append(contentsOf: $0) }
    try d.write(to: URL(fileURLWithPath: path))
}

func median(_ v: [Double]) -> Double { v.sorted()[v.count / 2] }
func span(_ v: [Double], _ digits: Int) -> String { "\(fmt(median(v), digits)) [\(fmt(v.min()!, digits))-\(fmt(v.max()!, digits))]" }

func runSay(_ args: Args, text: String) throws {
    let model = try args.loadModel()
    let request = try args.request(text: text, model: model)
    var pcm: [Float] = [], fragmentPCM: [[Float]] = []
    let t0 = Date()
    var firstAudio: Double?
    let reports = try model.synthesize(request, options: args.options) { fragment, chunk in
        firstAudio = firstAudio ?? Date().timeIntervalSince(t0)
        pcm += chunk
        while fragmentPCM.count <= fragment { fragmentPCM.append([]) }
        fragmentPCM[fragment] += chunk
    }
    let wall = Date().timeIntervalSince(t0)
    for (i, r) in reports.enumerated() {
        print("fragment \(i + 1): \(r.frames) frames, stopped \(r.stopped), prefill \(fmt(r.prefillSeconds * 1000, 1)) ms")
    }
    let audio = Double(pcm.count) / Double(BreezeTTSModel.sampleRate)
    let frames = reports.reduce(0) { $0 + $1.frames }
    print("\(fmt(audio, 2)) s of audio in \(fmt(wall, 2)) s wall (\(fmt(audio / wall, 2)) s of audio per s, \(fmt(Double(frames) / wall, 1)) frames/s), first audio after \(fmt((firstAudio ?? wall) * 1000, 0)) ms, peak \(fmt(Double(pcm.map(abs).max() ?? 0), 3))")
    try writeWav(pcm, to: args.out)
    print("wrote \(args.out)")
    if fragmentPCM.count > 1 {
        for (i, samples) in fragmentPCM.enumerated() {
            let path = args.out.hasSuffix(".wav") ? String(args.out.dropLast(4)) + ".frag\(i + 1).wav" : args.out + ".frag\(i + 1).wav"
            try writeWav(samples, to: path)
            print("wrote \(path)")
        }
    }
}

/// The fixed bench sentences: one unguided, one guided.
let benchText = "The garden will remain open until sunset, provided the weather stays clear, and the volunteers who tend the roses will stay to answer questions from visitors."
let benchCases: [(name: String, task: BreezeTask)] = [
    ("plain", .plain),
    ("design", .voiceDesign(instruction: "A calm middle-aged man with a warm, low voice, speaking at an even pace.")),
]

func runBench(_ args: Args) throws {
    let model = try args.loadModel()
    for (name, task) in benchCases {
        let request = try BreezeRequest(text: benchText, task: task)
        let guidance = task.recommendedGuidance
        let prompt = try model.prompt(fragment: benchText, of: request, guidance: guidance)
        print("bench \(name): guidance \(fmt(Double(guidance), 1)), \(prompt.positive.count) prompt rows, one warm-up run, then \(args.repeats) runs with seeds 1...\(args.repeats)")
        var rates: [Double] = [], firsts: [Double] = [], steps: [Double] = []
        for run in 0...args.repeats {
            var options = args.options
            options.guidance = nil
            options.seed = UInt64(run)
            var samples = 0
            var firstAudio: Double?
            let t0 = Date()
            let reports = try model.synthesize(request, options: options) { _, chunk in
                firstAudio = firstAudio ?? Date().timeIntervalSince(t0)
                samples += chunk.count
            }
            let wall = Date().timeIntervalSince(t0)
            guard run > 0 else { continue }
            let r = reports[0], audio = Double(samples) / Double(BreezeTTSModel.sampleRate)
            let frameWall = r.frameSeconds.reduce(0, +)
            rates.append(audio / wall)
            firsts.append((firstAudio ?? wall) * 1000)
            steps.append(Double(r.frames) / frameWall)
            print("run \(run): \(r.frames) frames, \(fmt(audio, 2)) s audio, wall \(fmt(wall, 3)) s, \(fmt(audio / wall, 2)) s audio/s, \(fmt(Double(r.frames) / frameWall, 1)) frames/s, first audio \(fmt(firsts.last!, 0)) ms, prefill \(fmt(r.prefillSeconds * 1000, 1)) ms, stopped \(r.stopped)")
        }
        print("summary \(name): audio/s \(span(rates, 2)), first audio ms \(span(firsts, 0)), frames/s \(span(steps, 1)); median [min-max] of \(args.repeats) runs")
    }
}

do {
    let args = try Args(CommandLine.arguments.dropFirst())
    if args.help {
        print(usage)
        exit(0)
    }
    switch args.mode {
    case .say(let text): try runSay(args, text: text)
    case .bench: try runBench(args)
    case .benchFrame:
        var frame = args.frame
        if let rounds = args.rounds { frame.rounds = rounds }
        try FrameBench.run(directory: try args.checkpoint(), options: frame)
    case .benchCodec:
        var codec = args.codec
        if let rounds = args.rounds { codec.rounds = rounds }
        if let frames = args.frames { codec.frames = frames }
        try CodecBench.run(directory: try args.checkpoint(), options: codec)
    case .benchGemv:
        try GemvBench.run(repeats: args.gemvRepeats, rounds: args.rounds ?? 5)
    case .benchPrompt:
        let lengths = ProcessInfo.processInfo.environment["BREEZE_BENCH_LENGTHS"]?.split(separator: ",").compactMap { Int($0) } ?? [60, 400]
        try PromptBench.run(directory: try args.checkpoint(), lengths: lengths, rounds: args.rounds ?? 7)
    case .parity(let caseDirectory):
        let report = try BreezeParity(weights: try args.checkpoint()).run(caseDirectory: caseDirectory)
        print(report.description)
        if !report.passed { exit(1) }
    case nil:
        print("usage: breeze-tts --say TEXT [--out FILE.wav] [--instruction TEXT] [--clone REF.wav --transcript TEXT] [--guidance S] [--seed N] | --bench [--repeat N] | --parity CASE_DIR, with [--weights DIR]; --help for more")
    }
} catch {
    FileHandle.standardError.write("breeze-tts: \(error)\n".data(using: .utf8)!)
    exit(2)
}
