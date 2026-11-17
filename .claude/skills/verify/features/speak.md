# Speak text

`breeze-tts --say "text"` turns text into speech and writes one WAV. Without voice flags the model speaks in its own voice, and `--instruction` designs a voice from a description. Long text is spoken fragment by fragment in one voice, and vocal-event tags are performed instead of read.

## Sub-features

- `speak-plain` speaks in the model's own voice and writes `out.wav`.
- `speak-design` speaks in a voice designed by `--instruction`, at guidance 4 unless `--guidance` says otherwise.
- `speak-chinese` speaks Chinese text, including a bracketed vocal event such as `[笑]`.
- `speak-fragments` splits long text, writes `out.fragN.wav` beside `out.wav`, and keeps fragment 1's voice in the later fragments.
- `speak-seed` makes output repeatable with `--seed`, and `--greedy` makes it independent of the seed.
- `speak-weights` finds the checkpoint from `--weights`, `BREEZE_WEIGHTS` or the Hugging Face cache, and names the missing files of a bad folder.
- `speak-errors` rejects an unknown argument, a flag with no value, and guidance that a plain request cannot take.

## How to get to it (user POV)

- Run `.build/release/breeze-tts --say "text"`. The WAV goes to `out.wav` in the working directory.
- Add `--out file.wav` to choose the file.
- Add `--instruction "description"` to design the voice.
- Add `--seed n`, `--guidance s`, `--greedy`, `--max-frames n` or `--fragment-length n`.
- Pass the checkpoint with `--weights dir`, or set `BREEZE_WEIGHTS`, or rely on the Hugging Face cache.
- Run `.build/release/breeze-tts --help` for every flag.

## Driving it with drive.sh

Preconditions:

- `$V/doctor.sh` exits 0 and reports `whisper`.

- **Plain.** Run `$V/drive.sh speak-plain -- --say "Thank you for calling. How can I help you today?"`. Exit 0. Stdout has `loaded`, one `fragment 1: N frames, stopped true` line with about 50 frames, a totals line with a `peak` between 0.1 and 1.0, and `wrote .../out.wav`. `wavstat.json` shows float, 1 channel, 24000 Hz and 0.08 s per frame. `transcript.txt` shows `wer 0.000`.
- **Design.** Run `$V/drive.sh speak-design -- --say "Welcome aboard. Your journey begins now." --instruction "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery."`. Same lines, about 40 frames, `wer 0.000`. Whether the voice fits the description needs a listener.
- **Chinese with an event.** Run `$V/drive.sh speak-chinese -- --say "[笑] 欢迎回来，今天过得怎么样？" --instruction "一位活泼开朗的年轻男性，声音明亮，语速适中。"`. `stopped true`, and `transcript.txt` shows `cer 0.000`: the transcript holds the words and not the tag. Whether the laugh is audible needs a listener.
- **Fragments.** Run `$V/drive.sh speak-fragments -- --say "The old lighthouse stood at the edge of the cliff for more than a century. Every evening the keeper climbed the narrow stairs to light the lamp, and every morning he wrote the weather in a worn leather book. When the storms came, ships far out at sea would search the darkness for its slow and steady beam." --instruction "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery." --fragment-length 25`. Three `fragment` lines, all `stopped true`, and `wrote` lines for `out.wav` and `out.frag1.wav` to `out.frag3.wav`. `transcript.txt` holds the three sentences in order with `wer 0.000`. To check the voice, run `tools/asr/speaker_similarity.py <dir>/out.frag*.wav` (see the helper table in SKILL.md for its interpreter and `CAMPPLUS_ONNX`): every pair scores about 0.8 or more, where fragments synthesized independently score about 0.6 and two different voices about 0.2.
- **Seed.** Run `$V/drive.sh speak-seed-a -- --say "Hello there." --seed 7`, then `$V/drive.sh speak-seed-b -- --say "Hello there." --seed 7`. `cmp` on the two `out.wav` files reports no difference. A third run with `--seed 8` differs.
- **Checkpoint.** Run `$V/drive.sh speak-weights-bad -- --say hi --weights /nonexistent`. Exit 2 with `bad checkpoint: /nonexistent lacks model.safetensors.index.json, ...` on stderr and no `out.wav`.
- **Bad arguments.** Run `$V/drive.sh speak-unknown -- --say hi --bogus`. Exit 2 with `bad input: unknown argument --bogus; see --help`. Run `$V/drive.sh speak-novalue -- --say`. Exit 2 with `--say needs a value`. Run `$V/drive.sh speak-guidance -- --say hi --guidance 4`. Exit 2 with `plain and clone requests take no guidance (got 4.0)`, and no `out.wav`.

## Gotchas

- `drive.sh` adds `--weights` whenever it finds a checkpoint, so it never exercises the tool's own search. To prove the Hugging Face cache path, run the binary directly with `BREEZE_WEIGHTS` unset and no `--weights`, and check that the `loaded` line names the snapshot.
- Whisper respells numbers. Keep digits out of proof sentences, or compare by ear.
- At the default `--fragment-length 60`, a paragraph under 60 words is one fragment. Lower it to exercise the split.
- A design request's voice comes from the instruction and the seed together, so two seeds give two speakers. Within one request the carry keeps one speaker.
