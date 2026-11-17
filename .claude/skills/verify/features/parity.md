# Golden parity

`breeze-tts --parity <case dir>` replays one case captured from the Python reference and compares the port's prompt, text-encoder and backbone tensors, logits and token streams with the captured ones. It is the check that the port still computes what the reference model computes. Greedy `--say` audio for the same case equals the reference's codec output.

## Sub-features

- `parity-case` passes one case: `plain_en`, `design_en`, `design_zh`, `clone_en`, `direction_en` or `long_text`.
- `parity-exit` exits 0 when every line passes and 1 when one fails.
- `parity-pcm` makes greedy `--say` audio equal the reference PCM for `plain_en`, `design_en` and `design_zh`.

## How to get to it (user POV)

- `breeze-tts --parity <case dir>`, with `--weights dir` or `BREEZE_WEIGHTS`.
- `breeze-tts --say "<case text>" [--instruction "<case instruction>"] --greedy`, then compare the WAV's samples with `<case dir>/codec/pcm_full.f32`.

## Driving it with drive.sh

Preconditions:

- `$V/doctor.sh` exits 0 and prints an `ok    goldens <dir>` line. The goldens are not published. With no goldens, report this feature as not run.
- `G` is the goldens directory from that line.

- **One case.** Run `$V/drive.sh parity-plain -- --parity "$G/plain_en"`. Exit 0. The last line of stdout is `summary plain_en: N tensors, worst cosine 0.99999..., N streams, 0 failed`.
- **Every case.** Run `for c in plain_en design_en design_zh clone_en direction_en long_text; do $V/drive.sh parity-$c -- --parity "$G/$c" | grep summary; done`. Every summary line ends `0 failed`, and every `exit.txt` holds 0.
- **Greedy PCM.** Run `$V/drive.sh pcm-plain -- --say "Thank you for calling. How can I help you today?" --greedy`. Stdout has `fragment 1: 49 frames, stopped true`. Then compare: `python3 -c "import numpy as n,sys; d=open(sys.argv[1],'rb').read(); g=n.frombuffer(d[44:],'<f4'); w=n.fromfile(sys.argv[2],'<f4'); print(len(g), len(w), abs(g-w).max())" <evidence dir>/out.wav "$G/plain_en/codec/pcm_full.f32"`. It prints `94080 94080` and a maximum difference near 2e-6. The design cases take their text and `--instruction` from `manifest.json` and match the same way, within 1e-5.

## Gotchas

- Compare greedy audio with `codec/pcm_full.f32`, not with the case's `fp32_greedy.wav`. The WAV is the reference's 2-frame streaming decode, written as 16-bit PCM, and the 2-frame decode drops one attention key per chunk after frame 72.
- `--say` with sampling (no `--greedy`) cannot match the reference sample for sample: the reference's draws come from torch's generator. The parity run replays the reference's stored uniforms instead.
- The clone cases cannot be compared through `--clone`, because the tool trims the clip's silence and so changes the prompt. `--parity` covers them.
- `codec_extra` and the `*.json` files in the goldens directory are not case directories.
