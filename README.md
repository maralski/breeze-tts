# BreezeTTS

BreezeTTS runs [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) by BreezeBlue on a Mac. It is a Swift and Metal port of the model's inference path, with no Python at run time. The package has two products:

- `BreezeTTS`, a library that turns text into 24 kHz mono audio and streams it as the model generates it, one 80 ms frame at a time.
- `breeze-tts`, a command-line tool that writes speech to a WAV file and runs benchmarks.

The port covers the four upstream tasks: plain text to speech in the model's own voice, voice design from a description, voice cloning from a reference clip, and voice direction (a cloned voice told how to speak). The model card lists English and Chinese.

> [!IMPORTANT]
> This repository's code is licensed under Apache-2.0. The Breeze TTS 2 weights are not. BreezeBlue publishes them under the BreezeBlue Research and Non-Commercial License Agreement, which permits research and non-commercial use only and grants no commercial rights. The agreement also covers the audio the model generates. Any commercial use of the weights or their outputs needs a separate written license from BreezeBlue (https://breezeblue.ai). The weights are not in this repository. See [NOTICE](NOTICE) and the [model's LICENSE](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE).

## Requirements

- A Mac with Apple silicon.
- macOS 14 or later.
- Xcode 16 or later (Swift 6 tools).
- About 8 GB of free disk space for the checkpoint (7.2 GB).

## Get the weights

The weights are not in this repository. Read the model's license, then download the checkpoint from Hugging Face:

```sh
pip install -U huggingface_hub
hf download BreezeBlue/Breeze-TTS-2
```

The directory must hold `model.safetensors.index.json`, its two shards, `tokenizer.json` and `audio_tokenizer/model.safetensors`. `BreezeCheckpoint(directory:).missingFiles` names the files a folder lacks, and loading a folder that lacks one fails with the same names.

## Build

```sh
swift build -c release
```

## Use the command-line tool

Pass the checkpoint directory with `--weights`, or set `BREEZE_WEIGHTS`. With neither, the tool uses the newest complete snapshot in the Hugging Face cache, which is where `hf download` puts it:

```sh
.build/release/breeze-tts --say "The garden stays open until sunset." --out garden.wav
```

Without voice options, the model speaks in its own voice. These options choose the voice:

| Option | Task |
| --- | --- |
| `--instruction "a calm middle-aged man with a warm, low voice"` | Voice design from a description. |
| `--clone ref.wav --transcript "..."` | Clone the voice in a reference clip. `--clone` trims silence from both ends of the clip first. |
| `--clone ref.wav --transcript "..." --instruction "speak slowly, almost in a whisper"` | Voice direction: the cloned voice, directed by the instruction. |

These options change sampling:

| Option | Effect |
| --- | --- |
| `--guidance S` | The guidance scale. The default is 4 for design and direction, and 1 for plain and clone requests, which accept no other value. |
| `--seed N` | The seed of the draws. The same text, options and seed give the same audio. |
| `--max-frames N` | The most frames per fragment, 80 ms each. The default is 1500. |
| `--fragment-length N` | The longest fragment in words or CJK characters. The default is 60. |
| `--greedy` | Argmax instead of sampling, for testing. |

Text longer than one fragment is spoken fragment by fragment. The tool then also writes each fragment as `FILE.fragN.wav`.

To measure speed, run `breeze-tts --bench` under `scripts/gpulock.py`. `breeze-tts --help` lists every option, including the parity check and the kernel benchmarks.

## Use the library

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/maralski/breeze-tts.git", from: "0.1.0")
```

Then load the model once and stream audio:

```swift
import BreezeTTS

let model = try BreezeTTSModel(directory: URL(fileURLWithPath: "/path/to/Breeze-TTS-2"))
let request = try BreezeRequest(
    text: "Welcome aboard. Your journey begins now.",
    task: .voiceDesign(instruction: "A warm, thoughtful young woman with a clear voice."))

for try await chunk in model.stream(request) {
    // chunk is [Float] PCM at BreezeTTSModel.sampleRate (24,000 Hz), mono: one 80 ms frame.
}
```

`BreezeCheckpoint.find()` returns the checkpoint `BREEZE_WEIGHTS` names, or the newest snapshot in the Hugging Face cache, or nil. Pass its `directory` to the model instead of a fixed path.

`stream(_:options:)` wraps the blocking `synthesize(_:options:onChunk:)`. Call that one directly to get a `BreezeGenerationReport` for each fragment, and stop it by throwing from the callback:

```swift
var options = BreezeSynthesisOptions()
options.seed = 7
let reports = try model.synthesize(request, options: options) { fragment, chunk in
    try Task.checkCancellation()
    // play or store chunk
}
let finished = reports.allSatisfy(\.stopped)
```

`BreezeGenerationReport.stopped` is `false` when a fragment ran to `BreezeSynthesisOptions.maxFrames` instead of ending by itself. Retry such a fragment with another seed.

### Long text keeps one voice

`synthesize` splits text longer than `BreezeSynthesisOptions.fragmentLength` (60 words or CJK characters by default) at sentence ends, and generates the fragments one after another. Plain and voice-design requests have no fixed voice, so each fragment would get a new speaker. To keep one voice, fragment 0's own audio codes and text become the reference voice for the later fragments. A design request continues as a voice direction with the same instruction and guidance, and a plain request continues as a clone. Clone and direction requests already carry a voice and run unchanged.

The carry holds the voice. We measured it on a 59-word paragraph in three fragments (15 to 21 s of speech) with five seeds, as the cosine similarity of CAM++ speaker embeddings between every two fragments of one paragraph:

| Request | With the carry | Each fragment on its own |
| --- | --- | --- |
| Voice design | 0.86 (0.81 to 0.88) | 0.59 (0.49 to 0.72) |
| Plain | 0.88 (0.85 to 0.93) | 0.38 (0.13 to 0.55) |

The table gives the median and the range of 15 pairs each. For scale, two different designed voices (a young woman and an old man) score 0.22. Whisper read every carried paragraph back with no word errors. Carrying a design as a clone instead of a direction scored 0.87 (0.80 to 0.90), no measurable difference, so the carry keeps the instruction and the guidance.

### Vocal events

Write a vocal event where it should happen. English text takes the parenthesized form, as in `"(sigh) It is good to hear you."`. Chinese text takes the bracketed form, as in `"[笑] 欢迎回来。"`. `BreezeVocalEvent.allCases` lists the four events the model card names: laugh, cough, clear throat and sigh. Other tags reach the model as written, and the model card does not promise it performs them.

### Clone a voice

Encode a reference clip once, then use it in any number of requests:

```swift
let (samples, rate) = try BreezeTTSModel.readMono(clipURL)
let voice = try model.encodeReference(samples: trimSilence(samples, sampleRate: rate), sampleRate: rate,
                                      transcript: "What the clip says, word for word.")
let cloned = try BreezeRequest(text: "Hello.", task: .voiceClone(voice))
let directed = try BreezeRequest(text: "Hello.", task: .voiceDirection(voice, instruction: "Whisper it."))
```

`BreezeReferenceVoice` is `Codable`, so you can store an encoded voice and reuse it.

- Use a clip of up to 20 seconds (`BreezeReferenceVoice.recommendedMaxSeconds`). The clip shares the backbone's 2048 positions with the prompt text and the generated audio. A 20 s clip uses about 335 of them and still leaves room for the full 1500 frames (120 s) of output. The encoder accepts clips of up to 40.96 s. A longer clip takes positions from the output: a 40 s clip uses about 585 and leaves room for about 117 s.
- Give the exact transcript of the clip. The model reads the transcript and then continues in the clip's voice.
- Trim silence from both ends with `trimSilence(_:sampleRate:)` first. `encodeReference(url:transcript:)` does not trim.
- Clips at other sample rates are resampled to 24 kHz with a windowed-sinc filter. The reference resamples with librosa, so the codes of such a clip are close to the reference's but not equal.
- Clone only voices you have the right to use. The model license forbids cloning a real person's voice without that person's explicit consent (section 5(c)), and using reference audio you lack the rights to (section 5(d)).

Clone requests accept no guidance other than 1. Direction requests take guidance 4 by default.

## Speed

Generation runs faster than real time on an M3 Max. These numbers come from `breeze-tts --bench` (two invocations of 5 seeded runs each, after one warm-up run) on an M3 Max with a 40-core GPU and 128 GB, macOS 27.0.1, in a release build, measured on 17 November 2026. Other processes (a virtual machine and system services) loaded the machine, at a load average of 35 to 60 when the runs ended:

| Request | Seconds of audio per second | Time to first audio | Frames per second |
| --- | --- | --- | --- |
| Plain, unguided | 1.43 and 1.50 (1.40 to 1.52) | 106 and 115 ms (100 to 123 ms) | 18.4 and 19.3 |
| Voice design, guidance 4 | 1.15 and 1.31 (1.11 to 1.35) | 208 and 189 ms (170 to 230 ms) | 14.9 and 17.1 |

Each cell gives the median of each invocation and the range over both. Time to first audio runs from the `synthesize` call to the first 80 ms chunk and includes the text encoding and the prefill. The first request after loading the model is slower, because it also allocates the frame loop: 186 to 289 ms to first audio over three runs of a plain sentence under the same load. At other times that day, the same benchmark measured as low as 1.24 (plain) and 1.08 (guided) seconds of audio per second.

The GPU time of the frame loop sets the speed. Earlier the same day, `breeze-tts --bench-frame` measured 44 ms per frame unguided and 49 ms guided, most of it in the depth decoder's 15 sequential steps per frame, and `breeze-tts --bench-codec --token-loop` measured 5 ms more per frame to decode it to audio. An end-to-end frame takes about 5% longer than that GPU work.

## Test

```sh
BREEZE_GOLDENS=/path/to/goldens swift test -c release
```

Tests that need files outside the repository are skipped when those files are missing:

- The checkpoint, from `BREEZE_WEIGHTS` or the Hugging Face cache, enables the tokenizer, weight, kernel-on-weights and model tests.
- `BREEZE_GOLDENS` enables the parity tests and the end-to-end check that greedy audio equals the reference's PCM. The goldens are tensors captured from the Python reference with `tools/reference/make_goldens.py`, and are not published.

The fixtures in `Tests/BreezeTTSTests/Fixtures` come from the upstream prompt templates and the Hugging Face tokenizer. `Fixtures/text/make_text_fixtures.py` regenerates them.

`tools/asr/transcribe.py` checks generated speech with Whisper and prints the word error rate. `tools/asr/speaker_similarity.py` compares speakers with CAM++ embeddings.

## License

BreezeTTS's code is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The Breeze TTS 2 weights, and the audio they generate, are under the BreezeBlue Research and Non-Commercial License Agreement: research and non-commercial use only, no commercial rights, and commercial use only under a separate written license from BreezeBlue. The audio tokenizer derives from [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by the Alibaba Qwen team (Apache-2.0). Parts of this code follow the upstream Breeze TTS 2 inference code (Apache-2.0) closely: the prompt templates, the sampling rules and the network definitions. NOTICE lists them with the upstream copyright notices.

This project is not affiliated with or endorsed by BreezeBlue.
