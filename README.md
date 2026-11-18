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

`--depth-weights int8` holds part of the model in 8 bits and generates faster. See [8-bit weights for the depth decoder](#8-bit-weights-for-the-depth-decoder) before you turn it on.

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

## 8-bit weights for the depth decoder

Each frame runs the depth decoder 15 times. Those passes read 10.0 GB of the 12.96 GB of weights a frame touches, and 90.6% of that is the three MLP projections of each of its 12 layers. With int8 weights those projections are held as 8-bit integers and the frame reads 8.63 GB.

The option is off by default. Turn it on when the model loads:

```swift
let model = try BreezeTTSModel(directory: directory, depthWeights: .int8)
```

```sh
.build/release/breeze-tts --say "The garden stays open until sunset." --depth-weights int8
```

The checkpoint on disk stays as it is. At load, each row of each projection is split into the 32 groups of weights that the 32 lanes of the kernel read, and each group gets one 16-bit scale, its largest weight over 127. Every weight is rounded to the nearest multiple of its scale. The result lives in GPU memory only and is never written to a file. Attention weights, the codebook heads, the embeddings and the backbone stay bf16. Quantizing the attention projections cost as much error as the whole MLP for 4.6% of the bytes. Loading takes about 0.2 s longer (1.6 s against 1.4 s, medians of five alternating loads with the files cached), and the model holds 289 MB less memory.

The bf16 path matches the reference's fp32 run stage by stage. The int8 path cannot. Its audio differs from the bf16 audio for the same seed, in the way two seeds differ.

### Speed

Both formats were loaded in one process and took turns, on the M3 Max described under [Speed](#speed), on 18 November 2026. The GPU column is the GPU time of a frame from `breeze-tts --bench-frame` with `BREEZE_FRAME_VARIANTS="default;int8"` (median of 9 rounds of 12 frames). The last column is `breeze-tts --bench --repeat 7 --compare-depth-weights`, end to end with the codec. Brackets hold the range over rounds or runs.

| Machine state | Request | GPU ms per frame, bf16 to int8 | Frames per second, bf16 to int8 |
| --- | --- | --- | --- |
| After ten idle minutes | Plain, unguided | 43.9 to 32.8 (11.1 ms less, 10.4 to 12.1 per round) | 19.9 to 25.9 (1.31 times, 1.30 to 1.35) |
| After ten idle minutes | Voice design, guidance 4 | 44.8 to 35.9 (8.9 ms less, 7.9 to 9.8) | 19.0 to 21.9 (1.17 times, 1.14 to 1.20) |
| After hours of generation | Plain, unguided | 52.0 to 39.9 (12.3 ms less, 6.1 to 15.1) | 16.9 to 21.4 (1.27 times, 1.25 to 1.29) |
| After hours of generation | Voice design, guidance 4 | 56.3 to 51.6 (6.7 ms less, 2.6 to 11.1) | 16.2 to 18.1 (1.13 times, 1.11 to 1.14) |

int8 was faster in every round and every run. Across the quality study below, which mixes every task, the runs on the int8 kernel produced 1.38 seconds of audio per second and the bf16 runs 1.20 to 1.25.

The frame reads 4.33 GB less, which at the frame's own rate of 295 GB/s would save 14.7 ms. An unguided frame gets three quarters of that. With one row of activations the int8 kernel still moves 255 to 341 GB/s of its smaller weights, so memory bandwidth still limits it. A guided frame gets less. Its two rows share each weight read, so every weight costs two multiply-adds and a conversion, and the kernel runs out of arithmetic at 178 to 261 GB/s. The rest of the frame is unchanged. The backbone step alone takes 9.4 ms of it.

### Quality

8 bits cannot reproduce bf16 weights, so "no loss" is a measured claim with limits, not a guarantee.

Two measurements were made, on 18 November 2026.

The first is the error in the depth decoder's logits, teacher-forced on the five reference cases (7,425 draws) and read where the sampler looks, after guidance, temperature 0.9 and top-k 50. Against the reference's fp32 run, the int8 weights move the sampling distribution by a total variation distance of 0.006 on average, and the most likely code agrees on 99.0% of draws. The reference's own bf16 arithmetic, which is how the model ships, moves it by 0.031 and agrees on 94.9%. The int8 error is about a fifth of that.

The second is a corpus study with a pass rule written down before any int8 audio was scored. It has 340 items in seven groups: plain English, voice design, cloning, direction, Chinese, vocal-event tags, and long text in three fragments. Each item ran four times, as bf16 and as int8 on each of two seeds. One of the two int8 runs used the int8 kernel. The other used the same int8 weights multiplied back out to bf16, as a control for the kernel. That gives 680 runs per format and 3.9 hours of audio in all. Whisper scored the words and CAM++ scored the speaker.

| Measure | bf16 | int8 |
| --- | --- | --- |
| Runs that did not reach end of speech | 0 of 680 | 0 of 680 |
| Runs with an error rate above 0.20 | 1 | 2 |
| Runs with an outlying length (frames per word) | 5 | 3 |
| Runs with an outlying speaker similarity (clone, direction, long text) | 4 of 280 | 2 of 280 |
| Mean word or character error rate | 0.0120 | 0.0109 |
| Mean speaker similarity | 0.831 | 0.834 |

No difference is significant, and every rule passed. Changing the format at a fixed seed changed a run about as much as changing the seed did: 6 and 7 items changed flag status between formats, and 8 between two bf16 seeds.

The study has limits:

- It cannot see a failure that happens less often than about once in 150 runs.
- Its judges are a speech recognizer and a speaker model. It does not measure prosody, fine timbre, or whether a vocal event is performed.
- It covers English and Mandarin, four reference voices made by the model itself, and text of up to three fragments.

So the option stays off by default.

Nobody has compared the two by ear. `tools/quant/study.py` reruns the study, and `tools/quant/depth_quant.py` reruns the logit measurements and the scheme sweep.

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
