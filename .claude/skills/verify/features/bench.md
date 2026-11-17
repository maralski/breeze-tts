# Benchmark

`breeze-tts --bench` speaks a fixed sentence plainly (unguided) and in a designed voice (guided at 4), one warm-up run and then several seeded runs each. For each run it prints the seconds of audio per second of wall time, the frames per second and the time to first audio, then a median and range per sentence.

## Sub-features

- `bench-run` prints a `bench <case>:` header, one `run N:` line per run, and one `summary <case>:` line, for the cases `plain` and `design`.
- `bench-repeat` sets the number of measured runs with `--repeat n`. Run `i` uses seed `i`.

## How to get to it (user POV)

- `scripts/gpulock.py -- .build/release/breeze-tts --bench`
- `scripts/gpulock.py -- .build/release/breeze-tts --bench --repeat 9`

## Driving it with drive.sh

Preconditions:

- `$V/doctor.sh` exits 0 with no `breeze-tts already running` warning.
- Nothing else heavy is using the GPU. Other worktrees' benchmarks wait on the same lock, but a `--say` or a test run that skips the lock still competes.

- **Run.** Run `$V/drive.sh bench-run -- --bench --repeat 5`. Exit 0. For each case, stdout has a header such as `bench plain: guidance 1.0, N prompt rows, one warm-up run, then 5 runs with seeds 1...5`, five lines such as `run 1: N frames, N s audio, wall N s, N s audio/s, N frames/s, first audio N ms, prefill N ms, stopped true`, and a line `summary plain: audio/s M [lo-hi], first audio ms M [lo-hi], frames/s M [lo-hi]; median [min-max] of 5 runs`. No WAV is written.
- **Read it.** Every `run` line must say `stopped true`. Seconds of audio per second above 1.0 means faster than real time. First audio counts from the request to the first 80 ms chunk and includes the prompt and the prefill, not the model load.

## Gotchas

- The warm-up run of each case is not printed or counted. A single `--say` reports the numbers of the first request after loading instead.
- The frame count changes with the seed, so seconds of audio per second is the comparable number, not wall time.
- A speed claim needs interleaved runs of both builds under the lock, with medians and ranges. Two separate `--bench` runs at different times differ by machine state more than most changes do.
- `--bench-frame`, `--bench-gemv`, `--bench-prompt` and `--bench-codec` are kernel microbenchmarks, not user features, and are not mapped here. `scripts/ab_bench.py` compares two builds on `--bench-frame`.
