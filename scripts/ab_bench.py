#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B of two breeze-tts builds under the GPU lock.

    scripts/ab_bench.py BASE_BINARY CANDIDATE_BINARY [--rounds 5] [--weights DIR, or BREEZE_WEIGHTS]

Each round runs `--bench-frame --rounds 3` on A then B (order alternates per round) while holding
the lock. Prints, per mode, the median GPU ms per frame of each side with its min-max range and B/A.
"""
import argparse, fcntl, os, re, statistics, subprocess

p = argparse.ArgumentParser()
p.add_argument("a"); p.add_argument("b")
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--weights", default=os.environ.get("BREEZE_WEIGHTS"))
args = p.parse_args()

FRAME = re.compile(r"frame (\w+): gpu ([\d.]+) ")
CHECK = re.compile(r"status clean true, codes in range true, codes identical in every run true")

def bench(binary):
    cmd = [binary, "--bench-frame", "--rounds", "3"] + (["--weights", args.weights] if args.weights else [])
    with open("/tmp/breeze-tts-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        out = subprocess.run(cmd, capture_output=True, text=True).stdout
    frames = {m.group(1): float(m.group(2)) for m in FRAME.finditer(out)}
    if not frames or len(CHECK.findall(out)) != len(frames):
        raise SystemExit(f"{binary}: the bench did not report clean, repeatable frames\n{out}")
    return frames

res = {"A": {}, "B": {}}
for r in range(args.rounds):
    order = [("A", args.a), ("B", args.b)] if r % 2 == 0 else [("B", args.b), ("A", args.a)]
    for name, binary in order:
        for mode, ms in bench(binary).items():
            res[name].setdefault(mode, []).append(ms)
for mode in res["A"]:
    a, b = res["A"][mode], res["B"][mode]
    print(f"{mode}: A {statistics.median(a):.2f} ms [{min(a):.2f}-{max(a):.2f}]  B {statistics.median(b):.2f} ms [{min(b):.2f}-{max(b):.2f}]  B/A {statistics.median(b) / statistics.median(a):.3f}  (n={len(a)})")
