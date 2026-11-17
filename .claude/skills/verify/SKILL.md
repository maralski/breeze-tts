---
name: verify
description: Drive the real breeze-tts command-line tool (the Swift and Metal port of Breeze TTS 2) and capture proof that it speaks correctly. Use after any change to Sources/ to prove behavior on the real binary and real weights: speech from text, voice design, voice cloning and direction, long text in fragments, golden parity, end-to-end benchmarks and error messages. Also use when asked to verify, demo or reproduce something in BreezeTTS.
---

# Verify BreezeTTS

BreezeTTS has two surfaces. The primary one is the `breeze-tts` command-line tool: it loads a 7.2 GB checkpoint, generates speech on the GPU and writes a WAV. The second is the `BreezeTTS` Swift library, which the tool wraps. Drive the library through the tool, and use `swift test` (see the end) for library-only API.

There is no server. Each run is one short process (about 2 s to load, then faster than real time), so "launch" means build once, then run the binary once per check.

All commands run from the repository root. `V` stands for `.claude/skills/verify/scripts`.

## Launch

1. Build the release binary. Debug builds are too slow to judge speed.

   ```sh
   swift build -c release
   ```

   Ready when it prints `Build complete!` and `.build/release/breeze-tts` exists. Rebuild after every source change. The doctor fails a stale binary.

2. Have a checkpoint. The helpers look at `BREEZE_WEIGHTS`, then the newest complete snapshot of `BreezeBlue/Breeze-TTS-2` in the Hugging Face cache, which is the order `breeze-tts` uses. The weights are not in the repository, and their license is research and non-commercial only. The README's "Get the weights" section has the download command. Do not download 7 GB without asking the user.

There is nothing to tear down: every run exits by itself.

## Doctor

```sh
.claude/skills/verify/scripts/doctor.sh
```

Read-only. It exits 0 when the machine is Apple silicon, the binary exists and is newer than every file in `Sources/` and `Package.swift`, and a complete checkpoint is found. It warns, without failing, when no goldens are found (parity cannot run), when speech recognition cannot run (no transcript evidence), and when another `breeze-tts` from this checkout is running (timings will be skewed). Run it first, and again whenever a result looks wrong.

## Drive

Every check goes through one helper, which runs the real binary and files the evidence:

```sh
.claude/skills/verify/scripts/drive.sh <name> -- <breeze-tts arguments>
```

`<name>` labels the evidence directory. Everything after `--` is passed to `breeze-tts` unchanged. The helper adds `--weights` and, for `--say`, `--out <evidence dir>/out.wav`, unless you pass them. It exits with the tool's exit code.

```sh
$V/drive.sh plain    -- --say "The garden stays open until sunset."
$V/drive.sh design   -- --say "Welcome aboard. Your journey begins now." --instruction "A warm, thoughtful young woman with a clear voice."
$V/drive.sh bad-cfg  -- --say hi --guidance 4        # expect exit 2
$V/drive.sh parity   -- --parity .audit/goldens/plain_en
$V/drive.sh bench    -- --bench --repeat 5
```

The stable handles are the flags and the lines the tool prints. `breeze-tts --help` prints every flag.

| Line on stdout | Meaning |
| --- | --- |
| `loaded <dir> in N s` | The checkpoint loaded. |
| `encoded the reference clip (N s, N frames) in N ms` | A `--clone` clip was trimmed and encoded. |
| `fragment N: N frames, stopped true, prefill N ms` | One line per text fragment. `stopped true` means the model ended the utterance itself. `stopped false` means it ran to `--max-frames`, which is a failure for normal text. |
| `N s of audio in N s wall (N s of audio per s, N frames/s), first audio after N ms, peak N` | Totals. `peak 0.000` is silence. The first-audio time includes the prompt and prefill, not the load. |
| `wrote <path>` | A WAV was written. Text that splits into several fragments also writes `out.fragN.wav`. |
| `summary <case>: N tensors, worst cosine N, N streams, N failed` | The last line of `--parity`. |
| `breeze-tts: bad input: ...` or `breeze-tts: bad checkpoint: ...` on stderr, exit 2 | A rejected command line or a bad weights folder. |

Exit codes are 0 for success, 1 for a failed parity case, and 2 for an error.

Two runs can go side by side. There are no ports and no shared files, and each evidence directory is named by time and `<name>`, so give concurrent runs different names. They share the GPU, so `drive.sh` runs every command under `scripts/gpulock.py`, which serializes GPU work across processes, including the benchmarks of other worktrees. Concurrent drives therefore wait for each other.

Pick the recipe for the feature you changed from [features/README.md](features/README.md). A proof that drives one entry point is incomplete when the feature file lists others your change touches.

## Evidence

Each run leaves a directory `.audit/verify/<UTC time>-<name>/`. The repository ignores `.audit/`. Set `BREEZE_VERIFY_DIR` to move the root.

| File | Content |
| --- | --- |
| `command.txt` | The exact command line. |
| `stdout.txt`, `stderr.txt`, `exit.txt` | What the tool printed and its exit code. |
| `out.wav`, `out.fragN.wav` | The audio: 32-bit float, mono, 24,000 Hz. |
| `wavstat.json` | Format, length, peak, RMS, non-finite sample count and silence at each end, per WAV. |
| `transcript.txt` | Whisper's reading of `out.wav` next to the `--say` text, with the word error rate (`wer`), or the character error rate (`cer`) when the text holds Chinese. Only for `--say` runs, and only when `tools/asr/transcribe.py` can run. Set `BREEZE_VERIFY_NO_ASR=1` to skip it. |

Proof standards:

- Drive the built binary with real weights. A passing unit test is not proof that the tool speaks.
- A WAV on disk proves little. Speech is proven by all of these: `stopped true` on every fragment, a `wavstat.json` with a plausible length (0.08 s per frame), a `peak` between roughly 0.1 and 1.0 and `nonfinite_samples` 0, and a `transcript.txt` whose error rate is near 0. Whisper respells numbers, so avoid digits in proof sentences.
- A transcript proves the words only. Claims about how it sounds (the voice, an emotion, a clone's likeness, a vocal event) need a listener. Send `out.wav` to the user and say that it was not judged by ear. `tools/asr/speaker_similarity.py` gives a number for "same speaker", not a judgment of quality.
- For a behavior change, capture a run from before and after the change, or a run of each branch of the behavior, not only the final state.
- For an error path, the evidence is `exit.txt` and `stderr.txt`, plus the absence of `out.wav`.
- Report the evidence directory path with every claim.

## Cleanup

- Processes: none outlive a run. If you interrupted one, kill the PID you started. Never run `pkill breeze-tts`: another worktree may be benchmarking. `doctor.sh` lists any `breeze-tts` still running from this checkout.
- Scratch: delete only files you made outside the evidence root, such as a trimmed reference clip in a temporary directory.
- Keep `.audit/verify/`. It is the proof. Leave `/tmp/breeze-tts-gpu.lock` alone, because other benchmark runs share it.

## Helpers

| Script | Use |
| --- | --- |
| `$V/doctor.sh` | The read-only readiness check above. |
| `$V/drive.sh <name> -- <args>` | Run `breeze-tts` and file the evidence. Run it with no arguments for its usage text. |
| `$V/wavstat.py <file.wav>` | Print one WAV's facts as JSON. Use it on a reference clip before cloning it. |
| `$V/common.sh` | Sourced by the two shell helpers. It holds the checkpoint and goldens search order. Do not run it directly. |
| `$BREEZE_ASR_PYTHON tools/asr/transcribe.py [--lang en\|zh] WAV TEXT ...` | Transcribe WAVs and score them against their text. `BREEZE_ASR_PYTHON` (default `python3`) must have openai-whisper and torch. |
| `tools/asr/speaker_similarity.py WAV WAV ...` | Cosine similarity of CAM++ speaker embeddings for every pair. Run it with a Python that has onnxruntime and torchaudio, and set `CAMPPLUS_ONNX` to a `campplus.onnx` (the Ming-3.0-tts checkpoint has one). |

## The library surface

For API that the tool does not reach (`BreezeCheckpoint.find(environment:)`, `stream` cancellation, `BreezeGenerationReport` fields, `trimSilence`), run the test suite the README documents:

```sh
BREEZE_GOLDENS=.audit/goldens swift test -c release
```

Tests that need the checkpoint or `BREEZE_GOLDENS` are skipped when they are missing, so a green run without them proves less than it appears to. Check the skip count.
