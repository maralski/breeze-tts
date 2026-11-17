# Clone a voice

`breeze-tts --say "text" --clone ref.wav --transcript "what the clip says"` speaks new text in the voice of a reference clip. Adding `--instruction` turns the clone into a voice direction: the cloned voice, told how to speak.

## Sub-features

- `clone-clip` clones the voice of a reference clip, given its exact transcript, after trimming silence from both ends.
- `clone-direction` directs the cloned voice with `--instruction`, at guidance 4 unless `--guidance` says otherwise.
- `clone-errors` rejects `--clone` without `--transcript`, `--transcript` without `--clone`, and guidance other than 1 on a plain clone.

## How to get to it (user POV)

- `breeze-tts --say "text" --clone ref.wav --transcript "what the clip says"`
- `breeze-tts --say "text" --clone ref.wav --transcript "..." --instruction "speak slowly, almost in a whisper"`
- The clip may be any format AVFoundation reads, at any sample rate.

## Driving it with drive.sh

Preconditions:

- `$V/doctor.sh` exits 0 and reports `whisper`.
- A reference clip whose words you know exactly. The goldens hold one: `REF="$G/ref_clip_en.wav"`, whose transcript is `Welcome aboard. Your journey begins now.` Without goldens, make one with the tool: run `$V/drive.sh clone-ref -- --say "Welcome aboard. Your journey begins now." --instruction "A warm, thoughtful young woman with a clear voice."` and use its `out.wav`.

- **Clone a clip.** Run `$V/drive.sh clone-clip -- --say "Please keep your seatbelt fastened until the sign turns off." --clone "$REF" --transcript "Welcome aboard. Your journey begins now."`. Exit 0. Stdout has `encoded the reference clip (N s, N frames) in N ms`, then `fragment 1: N frames, stopped true`. `transcript.txt` shows `wer 0.000` and holds none of the reference's words. Compare the voices with `tools/asr/speaker_similarity.py "$REF" <dir>/out.wav`.
- **Direction.** Run the same command as `clone-direction` with `--instruction "Speak slowly and softly, almost in a whisper."` added. Exit 0, `stopped true`, `wer 0.000`. Whether it sounds softer needs a listener.
- **Missing transcript.** Run `$V/drive.sh clone-no-transcript -- --say hi --clone "$REF"`. Exit 2 with `bad input: --clone needs --transcript` and no `out.wav`.
- **Guidance on a clone.** Run `$V/drive.sh clone-guidance -- --say hi --clone "$REF" --transcript "Welcome aboard. Your journey begins now." --guidance 4`. Exit 2 with `plain and clone requests take no guidance`.
- **Proof of likeness.** The transcript proves the words, not the voice. Send the reference clip and the clone's `out.wav` to the user together, and say the likeness was not judged by ear.

## Gotchas

- The transcript must be the clip's exact words. The model reads the transcript and then continues in the clip's voice, so a wrong transcript gives wrong or garbled words while still exiting 0.
- `--clone` trims silence from both ends. The library call `encodeReference(url:transcript:)` does not trim.
- Keep the clip under 20 s. A longer clip takes backbone positions from the output, and the encoder refuses clips over 40.96 s.
- The model license forbids cloning a real person's voice without that person's explicit consent. Use clips you have the right to use.
- A clip at a rate other than 24 kHz is resampled with a different filter than the reference's, so its codes differ slightly from the Python reference's.
