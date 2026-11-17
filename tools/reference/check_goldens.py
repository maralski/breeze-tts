# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Reload the goldens, check them against their manifests, and compare precisions.

Environment: any Python 3 with numpy (the repo venv works). Optional env var
BREEZE_GOLDENS (default <repo>/.audit/goldens). Exits non-zero when a file does
not match its manifest entry or a float tensor holds NaN or inf.

For each tensor present in both the bf16 and fp32 runs it prints the cosine
over the whole tensor, the worst cosine over rows of the first axis (frames,
positions, or channels), and the max absolute difference.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

GOLDENS = Path(
    os.environ.get("BREEZE_GOLDENS", Path(__file__).resolve().parents[2] / ".audit" / "goldens")
)


def load(case_dir: Path, entry: dict) -> np.ndarray:
    path = case_dir / entry["file"]
    dtype = "<f4" if entry["dtype"] == "float32" else "<i4"
    count = int(np.prod(entry["shape"])) if entry["shape"] else 1
    if path.stat().st_size != count * 4:
        raise ValueError(f"{path}: {path.stat().st_size} bytes, manifest shape {entry['shape']}")
    return np.fromfile(path, dtype=dtype).reshape(entry["shape"])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom else float(np.array_equal(a, b))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def inverse_cdf(p: np.ndarray, u: np.float32) -> int:
    cdf = np.cumsum(p.astype(np.float64))
    hit = np.nonzero(cdf > float(u) * cdf[-1])[0]
    return int(hit[0]) if len(hit) else int(np.nonzero(p > 0)[0][-1])


def replay_backbone(logits: np.ndarray, history: np.ndarray, u) -> int:
    x = logits.astype(np.float32).copy()
    seen = np.unique(history)
    x[seen] = np.where(x[seen] > 0, x[seen] / np.float32(1.1), x[seen] * np.float32(1.1))
    x[2048:2051] = -np.inf
    if u is None:
        return int(np.argmax(x))
    x = x / np.float32(0.9)
    x[x < np.sort(x)[-50]] = -np.inf
    return inverse_cdf(softmax(x), u)


def replay_depth(logits: np.ndarray, u) -> int:
    g = logits.astype(np.float32).copy()
    g[2048:2051] = -np.inf
    if u is None:
        return int(np.argmax(g))
    x = g / np.float32(0.9)
    x[x < np.sort(x)[-50]] = -np.inf
    return inverse_cdf(softmax(x), u)


def replay_stream(run: dict[str, np.ndarray], mode: str, eos: bool) -> int:
    """Count tokens the documented sampler rules fail to reproduce from the stored tensors."""
    codes = run[f"{mode}_codes"]
    sampled = mode == "sampled"
    want = list(codes[:, 0]) + ([2051] if eos else [])
    misses = 0
    for m, target in enumerate(want):
        u = run[f"{mode}_u_backbone"][m] if sampled else None
        misses += replay_backbone(run[f"{mode}_backbone_logits_cfg"][m], codes[:m, 0], u) != target
    for f in range(len(codes)):
        for c in range(15):
            u = run[f"{mode}_u_depth"][f, c] if sampled else None
            misses += replay_depth(run[f"{mode}_depth_logits_cfg"][f, c], u) != codes[f, c + 1]
    return misses


def main() -> int:
    failures = 0
    rows = []
    for case_dir in sorted(p for p in GOLDENS.iterdir() if (p / "manifest.json").exists()):
        manifest = json.loads((case_dir / "manifest.json").read_text())
        sections = {f"runs.{k}": v["tensors"] for k, v in manifest.get("runs", {}).items()}
        for key in ("codec", "encoder"):
            if key in manifest:
                sections[key] = manifest[key]["tensors"]
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for section, tensors in sections.items():
            for name, entry in tensors.items():
                try:
                    arr = load(case_dir, entry)
                except ValueError as exc:
                    print("FAIL", exc)
                    failures += 1
                    continue
                if entry["dtype"] == "float32" and not np.isfinite(arr).all():
                    print(f"FAIL {case_dir.name}/{entry['file']}: {np.size(arr) - np.isfinite(arr).sum()} non-finite")
                    failures += 1
                arrays.setdefault(section, {})[name] = arr
        bf16, fp32 = arrays.get("runs.bf16", {}), arrays.get("runs.fp32", {})
        if fp32:
            for mode in ("greedy", "sampled"):
                eos = manifest["runs"]["fp32"][mode]["stopped_on_eos"]
                misses = replay_stream(fp32, mode, eos)
                print(f"{case_dir.name}: fp32 {mode} sampler replay from stored tensors, {misses} mismatched tokens")
                failures += misses > 0
            if fp32["greedy_codes"].shape == fp32["sampled_codes"].shape and np.array_equal(
                fp32["greedy_codes"], fp32["sampled_codes"]
            ):
                print(f"FAIL {case_dir.name}: sampled codes equal greedy codes")
                failures += 1
            if fp32["sampled_u_backbone"].max() >= 1 or fp32["sampled_u_depth"].max() >= 1:
                print(f"FAIL {case_dir.name}: a stored uniform rounds to 1.0")
                failures += 1
        for name in sorted(set(bf16) & set(fp32)):
            a, b = bf16[name], fp32[name]
            if a.dtype != np.float32 or a.shape != b.shape:
                same = a.shape == b.shape and np.array_equal(a, b)
                rows.append((case_dir.name, name, str(list(a.shape)), "equal" if same else "DIFFER", "", ""))
                continue
            flat = a.reshape(a.shape[0], -1) if a.ndim > 1 else a[None]
            ref = b.reshape(b.shape[0], -1) if b.ndim > 1 else b[None]
            worst = min(cosine(x, y) for x, y in zip(flat, ref))
            rows.append(
                (
                    case_dir.name,
                    name,
                    str(list(a.shape)),
                    f"{cosine(a, b):.6f}",
                    f"{worst:.6f}",
                    f"{np.abs(a - b).max():.3g}",
                )
            )
        print(f"{case_dir.name}: " + ", ".join(f"{s} {len(t)} tensors" for s, t in arrays.items()))
    header = ("case", "tensor", "shape", "cos", "min row cos", "max abs")
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    print()
    for r in [header, *rows]:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
