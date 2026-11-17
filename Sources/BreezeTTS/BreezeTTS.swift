// Copyright 2026 Peter Marelas
// SPDX-License-Identifier: Apache-2.0
import AVFoundation
import Foundation

public struct SynthesisOptions: Sendable {
    /// Classifier-free guidance scale. Nil uses the task's `recommendedGuidance`: 4 for voice
    /// design and voice direction, as the upstream README runs them, and 1 (none) for plain and
    /// clone requests, which have no negative prompt and reject any other value.
    public var guidance: Float?
    /// Seeds the uniform draws. The same request, options and seed give the same audio.
    public var seed: UInt64 = 0
    /// Most frames (80 ms each) one fragment may generate; the reference's `max_new_tokens`.
    /// Generation also stops where the prompt and the frames fill the backbone's 2048 positions.
    public var maxFrames = 1500
    /// The reference's sampling settings, on the backbone and the depth decoder alike.
    public var temperature: Float = 0.9
    public var topK = 50
    /// Applied to the backbone's draws from the second frame on, as the reference applies it.
    public var repetitionPenalty: Float = 1.1
    /// Argmax at every draw instead of sampling. Deterministic without a seed; the reference
    /// always samples, so use it for testing, not for listening.
    public var greedy = false
    /// Longest text fragment synthesized as one utterance, in words and CJK characters (see
    /// `BreezeRequest.fragments(maxLength:)`).
    public var fragmentLength = BreezeRequest.defaultFragmentLength
    /// Replaces the seeded uniforms (parity runs replay the reference's).
    var uniforms: UniformSource?

    public init() {}

    func frameSettings(guidance: Float?) -> FrameSettings {
        let s = SamplerSettings(temperature: temperature, topK: topK, repetitionPenalty: repetitionPenalty, greedy: greedy)
        return FrameSettings(guidance: guidance, backbone: s, depth: s)
    }

    func validate() throws {
        guard maxFrames >= 1 else { throw BreezeTTSError.badInput("maxFrames must be at least 1, not \(maxFrames)") }
        guard temperature.isFinite, temperature > 0 else { throw BreezeTTSError.badInput("temperature must be a positive number, not \(temperature)") }
        guard (1...BreezeReferenceVoice.codebookSize).contains(topK) else {
            throw BreezeTTSError.badInput("topK must lie in 1...\(BreezeReferenceVoice.codebookSize), not \(topK)")
        }
        guard repetitionPenalty.isFinite, repetitionPenalty > 0 else {
            throw BreezeTTSError.badInput("repetitionPenalty must be a positive number, not \(repetitionPenalty)")
        }
        guard fragmentLength >= 1 else { throw BreezeTTSError.badInput("fragmentLength must be at least 1, not \(fragmentLength)") }
    }
}

/// What one fragment took.
public struct GenerationReport: Sendable {
    /// Frames generated, 80 ms of audio each. The end-of-speech frame is not counted.
    public var frames: Int
    /// True when the model ended the utterance; false when it ran to `maxFrames` or the
    /// backbone's position limit.
    public var stopped: Bool
    /// Text encoding and the backbone prefill, wall clock.
    public var prefillSeconds: Double
    /// Per frame, wall clock from the previous frame (or the prefill) to its codes.
    public var frameSeconds: [Double]
    /// Wall clock from the start of the fragment (prefill included) to its first audio; nil
    /// when it generated no frames.
    public var firstAudioSeconds: Double?

    init(_ r: SynthesisReport) {
        frames = r.frames
        stopped = r.stoppedOnEndOfSpeech
        prefillSeconds = r.prefillSeconds
        frameSeconds = r.frameSeconds
        firstAudioSeconds = r.firstAudioSeconds
    }
}

/// Breeze TTS 2 on the GPU. Audio is 24 kHz mono Float32, produced while the model generates.
/// One request runs at a time; concurrent calls wait their turn.
public final class BreezeTTSModel: @unchecked Sendable {
    public static let sampleRate = 24_000

    let synth: Synthesizer
    let decoder: CodecDecoder
    let tokenizer: BreezeTokenizer
    private let directory: URL
    private let lock = NSLock()
    private var encoderCache: CodecEncoder?
    private let encoderLock = NSLock()

    /// `directory` holds the Hugging Face snapshot of `BreezeCheckpoint.repository`.
    /// - Throws: `BreezeTTSError.badCheckpoint` naming the files the folder lacks.
    ///   `BreezeCheckpoint` makes the same check without loading.
    public init(directory: URL) throws {
        try BreezeCheckpoint(directory: directory).require(BreezeCheckpoint.synthesisFiles)
        self.directory = directory
        synth = try Synthesizer(directory: directory)
        decoder = try CodecDecoder(directory: directory, ctx: synth.ctx)
        tokenizer = try BreezeTokenizer(tokenizerJSON: directory.appendingPathComponent("tokenizer.json"))
        try synth.ctx.warmUpPipelines()
    }

    // MARK: synthesis

    /// Streams PCM chunks for `request` as they are decoded, one 80 ms frame at a time.
    /// Cancelling the consuming task stops generation at the next frame.
    public func stream(_ request: BreezeRequest, options: SynthesisOptions = .init()) -> AsyncThrowingStream<[Float], Error> {
        AsyncThrowingStream { continuation in
            let task = Task.detached(priority: .userInitiated) {
                do {
                    _ = try self.synthesize(request, options: options) { _, chunk in
                        try Task.checkCancellation()
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Synthesizes every fragment of `request` in order, delivering each frame's PCM with its
    /// fragment index as the GPU finishes it. Throwing from `onChunk` stops generation and
    /// rethrows the error.
    ///
    /// Plain and voice-design requests have no voice of their own beyond the prompt, so each
    /// fragment would get a new speaker. When such a request splits, fragment 0's own codes and
    /// text become the reference voice for the rest: a design request continues as a voice
    /// direction (the instruction and the guidance stay), a plain one as a clone.
    public func synthesize(_ request: BreezeRequest, options: SynthesisOptions = .init(), onChunk: (Int, [Float]) throws -> Void) throws -> [GenerationReport] {
        try options.validate()
        let fragments = request.fragments(maxLength: options.fragmentLength)
        lock.lock()
        defer { lock.unlock() }
        var request = request
        var reports: [GenerationReport] = []
        for (index, fragment) in fragments.enumerated() {
            let prompt = try BreezePrompt(fragment: fragment, of: request, guidance: options.guidance ?? request.task.recommendedGuidance, tokenizer: tokenizer)
            let carries = index == 0 && fragments.count > 1 && request.task.reference == nil
            var codes: [[Int16]] = [], pcm: [Float] = []
            let report = try synth.run(prompt, options: options, decoder: decoder, onAudio: { chunk in
                if carries { pcm += chunk }
                try onChunk(index, chunk)
            }, onFrame: { frame in
                if carries { codes.append(frame.map { Int16($0) }) }
            })
            reports.append(GenerationReport(report))
            if carries, let task = try Self.carriedTask(request.task, fragment: fragment, codes: codes, pcm: pcm) {
                request = try BreezeRequest(text: request.text, task: task)
            }
        }
        return reports
    }

    /// The task later fragments run as, with fragment 0's frames up to its trailing silence as
    /// the reference voice. Nil leaves the task as it is: fragment 0 produced no speech, or so
    /// much that the clip would crowd the backbone's positions.
    static func carriedTask(_ task: BreezeTask, fragment: String, codes: [[Int16]], pcm: [Float]) throws -> BreezeTask? {
        let spoken = trimTrailingSilence(pcm, sampleRate: sampleRate).count
        let frames = min(codes.count, (spoken + CodecShape.samplesPerFrame - 1) / CodecShape.samplesPerFrame)
        guard frames > 0, Double(frames) / BreezeReferenceVoice.framesPerSecond <= 2 * BreezeReferenceVoice.recommendedMaxSeconds else { return nil }
        let voice = try BreezeReferenceVoice(transcript: fragment, codes: Array(codes.prefix(frames)))
        switch task {
        case .plain: return .voiceClone(voice)
        case .voiceDesign(let instruction): return .voiceDirection(voice, instruction: instruction)
        case .voiceClone, .voiceDirection: return nil
        }
    }

    /// The fragments `request` is spoken in, in order.
    public func fragments(for request: BreezeRequest, fragmentLength: Int = SynthesisOptions().fragmentLength) -> [String] {
        request.fragments(maxLength: fragmentLength)
    }

    /// The backbone prompt for `fragment` under `request`'s task, as `synthesize` builds it.
    /// - Throws: `BreezeTTSError.badInput` for guidance the task rejects, or a prompt that leaves
    ///   the backbone no room to generate.
    public func prompt(fragment: String, of request: BreezeRequest, guidance: Float? = nil) throws -> BreezePrompt {
        try BreezePrompt(fragment: fragment, of: request, guidance: guidance ?? request.task.recommendedGuidance, tokenizer: tokenizer)
    }

    // MARK: reference voices

    /// Encodes a reference clip for cloning: mono samples at any rate, resampled to 24 kHz and
    /// run through the codec's encoder. Persist the result and reuse it. Trim the clip with
    /// `trimSilence` first. A clip may be up to 40.96 s, but keep it under
    /// `BreezeReferenceVoice.recommendedMaxSeconds`.
    public func encodeReference(samples: [Float], sampleRate: Int, transcript: String) throws -> BreezeReferenceVoice {
        guard !samples.isEmpty else { throw BreezeTTSError.badInput("reference audio is empty") }
        let codes = try codecEncoder().encode(samples: samples, sampleRate: sampleRate)
        return try BreezeReferenceVoice(transcript: transcript, codes: codes)
    }

    /// Decodes `url` with AVFoundation (any format it reads), mixes to mono, and encodes it.
    /// It does not trim silence.
    public func encodeReference(url: URL, transcript: String) throws -> BreezeReferenceVoice {
        let (samples, rate) = try Self.readMono(url)
        return try encodeReference(samples: samples, sampleRate: rate, transcript: transcript)
    }

    private func codecEncoder() throws -> CodecEncoder {
        encoderLock.lock()
        defer { encoderLock.unlock() }
        if let encoderCache { return encoderCache }
        try BreezeCheckpoint(directory: directory).require(BreezeCheckpoint.cloningFiles)
        let encoder = try CodecEncoder(directory: directory, ctx: synth.ctx)
        encoderCache = encoder
        return encoder
    }

    /// The file's samples mixed to mono at its own rate, for `trimSilence` before
    /// `encodeReference(samples:sampleRate:transcript:)`.
    public static func readMono(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw BreezeTTSError.badInput("\(url.lastPathComponent) holds no audio")
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else { throw BreezeTTSError.badInput("\(url.lastPathComponent) is not decodable to float PCM") }
        let count = Int(buffer.frameLength), channelCount = Int(format.channelCount)
        var mono = [Float](repeating: 0, count: count)
        for c in 0..<channelCount {
            let data = channels[c]
            for i in 0..<count { mono[i] += data[i] }
        }
        if channelCount > 1 { for i in 0..<count { mono[i] /= Float(channelCount) } }
        return (mono, Int(format.sampleRate))
    }
}
