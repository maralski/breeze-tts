#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Corpus study of the int8 depth-decoder weights against bf16: does speech get worse?

    python3 tools/quant/study.py plan --out DIR      # the item list and the reference voices (GPU)
    python3 tools/quant/study.py run --out DIR       # every arm through breeze-tts --batch (GPU)
    python3 tools/quant/study.py asr --out DIR       # Whisper; a Python with openai-whisper
    .venv/bin/python tools/quant/study.py spk --out DIR   # CAM++; needs onnxruntime, CAMPPLUS_ONNX
    python3 tools/quant/study.py gate --out DIR      # the decision rule, printed and saved

Every step resumes: it skips what DIR already holds. Build the tool first (swift build -c release).
BREEZE_TOOL names another binary. `run` records the binary's SHA-256 in DIR/tool.txt and refuses
to add runs from a different one, so a rebuild cannot change the model halfway through a study.

Four arms, two formats by two seeds per item:
    A  bf16, seed s          B  int8 kernel, seed s
    C  bf16, seed s'         R  int8 weights dequantized onto the bf16 kernels, seed s'
A and C differ only by seed: how far two runs of one model differ is the noise floor that the
difference between A and B is read against. Same-seed arms stop being paired at the first draw
that differs, so the comparison is of rates and means, not of clips.

The strata, flags and decision rule were written to .audit/quantization.tsv (row GATE) before
any int8 audio was scored. `gate` applies exactly that rule.
"""
import argparse, hashlib, json, math, os, random, re, statistics, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = Path(os.environ.get("BREEZE_TOOL", ROOT / ".build/release/breeze-tts"))
ARMS = {"A": ("bf16", 0, {}), "B": ("int8", 0, {}), "C": ("bf16", 1, {}), "R": ("int8", 1, {"BREEZE_INT8_REFERENCE": "1"})}
BF16, INT8 = ("A", "C"), ("B", "R")
PAIRS = (("A", "B"), ("C", "R"))
SEED0, SEED_STEP = 1000, 100000
SPEAKER_STRATA = ("clone_en", "direction_en", "long")
TAG = re.compile(r"\([^()]{1,32}\)|\[[^\[\]]{1,32}\]")


def units(text, lang):
    text = TAG.sub(" ", text).lower()
    return len([c for c in text if c.isalnum()]) if lang == "zh" else len(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text))


def build_items():
    c = json.loads((Path(__file__).parent / "corpus.json").read_text())
    items = []

    def add(stratum, lang, text, instruction=None, voice=None):
        n = units(text, lang)
        cap = 900 if stratum == "long" else max(150, min(900, n * (10 if lang == "zh" else 15)))
        items.append(dict(id=f"{len(items):03d}_{stratum}", stratum=stratum, lang=lang, text=text, instruction=instruction, voice=voice,
                          units=n, max_frames=cap, seed=SEED0 + len(items)))

    en, di, dr, voices = c["sentences_en"], c["design_instructions_en"], c["direction_instructions_en"], [v["id"] for v in c["voices"]]
    for rep in range(2):
        for i, s in enumerate(en):
            add("plain_en", "en", s)
        for i, s in enumerate(en):
            add("design_en", "en", s, instruction=di[(i + rep) % len(di)])
        for i, s in enumerate(en[:25]):
            add("clone_en", "en", s, voice=voices[(i + rep) % len(voices)])
        for i, s in enumerate(en[5:]):
            add("direction_en", "en", s, instruction=dr[(i + rep) % len(dr)], voice=voices[(i + 2 * rep + 1) % len(voices)])
        for i, s in enumerate(c["sentences_zh"]):
            add("zh", "zh", s, instruction=c["design_instructions_zh"][i % 3] if rep else None)
        for i, s in enumerate(c["tagged_en"]):
            add("tags", "en", s, instruction=di[i % len(di)] if rep else None)
        for i, s in enumerate(c["tagged_zh"]):
            add("tags", "zh", s, instruction=c["design_instructions_zh"][i % 3] if rep else None)
        for i, s in enumerate(c["long_en"]):
            add("long", "en", s)
            add("long", "en", s, instruction=di[(i + rep) % len(di)])
    return items, c["voices"]


def gpu(cmd, env=None, **kw):
    return subprocess.run([sys.executable, str(ROOT / "scripts/gpulock.py"), "--", *map(str, cmd)], env={**os.environ, **(env or {})}, **kw)


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def cmd_plan(args):
    out = Path(args.out).resolve()
    (out / "voices").mkdir(parents=True, exist_ok=True)
    items, voices = build_items()
    todo = [dict(id=v["id"], text=v["transcript"], instruction=v["instruction"], seed=v["seed"], out=str(out / "voices" / f"{v['id']}.wav"))
            for v in voices if not (out / "voices" / f"{v['id']}.wav").exists()]
    if todo:
        batch = out / "voices" / "batch.jsonl"
        batch.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in todo))
        gpu([TOOL, "--batch", batch, "--depth-weights", "bf16"], check=True)
    (out / "voices.json").write_text(json.dumps(voices, ensure_ascii=False, indent=1))
    (out / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    counts = {}
    for i in items:
        counts[i["stratum"]] = counts.get(i["stratum"], 0) + 1
    print(f"{len(items)} items: {counts}; reference voices in {out / 'voices'}")


def cmd_run(args):
    out = Path(args.out).resolve()
    items = read_jsonl(out / "items.jsonl")
    voices = {v["id"]: v for v in json.loads((out / "voices.json").read_text())}
    digest = hashlib.sha256(TOOL.read_bytes()).hexdigest()
    pinned = out / "tool.txt"
    if not pinned.exists():
        pinned.write_text(digest + "\n")
    elif pinned.read_text().strip() != digest:
        sys.exit(f"{TOOL} is not the binary this study started with (see {pinned})")
    chunks = [items[i:i + args.chunk] for i in range(0, len(items), args.chunk)]
    for n, chunk in enumerate(chunks):
        for arm in args.arms:
            fmt, seed_index, env = ARMS[arm]
            d = out / arm
            d.mkdir(exist_ok=True)
            done = {r["id"] for r in read_jsonl(d / "results.jsonl")}
            todo = []
            for it in chunk:
                if it["id"] in done:
                    continue
                t = dict(id=it["id"], text=it["text"], seed=it["seed"] + seed_index * SEED_STEP, max_frames=it["max_frames"], out=str(d / f"{it['id']}.wav"))
                if it["instruction"]:
                    t["instruction"] = it["instruction"]
                if it["voice"]:
                    t["clone"] = str(out / "voices" / f"{it['voice']}.wav")
                    t["transcript"] = voices[it["voice"]]["transcript"]
                todo.append(t)
            if not todo:
                continue
            batch = d / "batch.jsonl"
            batch.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in todo))
            p = gpu([TOOL, "--batch", batch, "--depth-weights", fmt], env=env, capture_output=True, text=True)
            lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
            with open(d / "results.jsonl", "a") as f:
                f.write("".join(l + "\n" for l in lines))
            print(f"chunk {n + 1}/{len(chunks)} arm {arm}: {len(lines)} of {len(todo)} runs", flush=True)
            if p.returncode != 0 or len(lines) != len(todo):
                sys.exit(f"arm {arm} failed: {p.stderr[-2000:]}")


def cmd_asr(args):
    sys.path.insert(0, str(ROOT / "tools/asr"))
    import transcribe, whisper
    out = Path(args.out).resolve()
    items = read_jsonl(out / "items.jsonl")
    jobs = [(arm, it) for arm in args.arms for it in items] + ([("voices", v) for v in json.loads((out / "voices.json").read_text())] if args.voices else [])
    models = {}
    for arm, it in jobs:
        d = out / arm
        done = {r["id"] for r in read_jsonl(d / "asr.jsonl")}
        if it["id"] in done or not (d / f"{it['id']}.wav").exists():
            continue
        lang = it.get("lang", "en")
        text = it.get("text") or it["transcript"]
        if lang not in models:
            models[lang] = transcribe.load(lang)
        options = dict(language=lang, fp16=False, temperature=0.0, condition_on_previous_text=False)
        if lang == "zh":
            options["initial_prompt"] = "以下是普通话的句子。"
        got = whisper.transcribe(models[lang], str(d / f"{it['id']}.wav"), **options)["text"].strip()
        err = transcribe.error_rate(transcribe.tokens(text, lang), transcribe.tokens(got, lang))
        with open(d / "asr.jsonl", "a") as f:
            f.write(json.dumps(dict(id=it["id"], transcript=got, err=err), ensure_ascii=False) + "\n")
        print(arm, it["id"], f"{err:.3f}", flush=True)


def cmd_spk(args):
    sys.path.insert(0, str(ROOT / "tools/asr"))
    import onnxruntime, speaker_similarity as ss
    out = Path(args.out).resolve()
    session = onnxruntime.InferenceSession(os.environ["CAMPPLUS_ONNX"], providers=["CPUExecutionProvider"])
    cache = {}

    def emb(path):
        if path not in cache:
            try:
                cache[path] = ss.embed(session, str(path)) if ss.read(str(path)).numel() >= 8000 else None
            except Exception:
                cache[path] = None
        return cache[path]

    for arm in args.arms:
        d = out / arm
        done = {r["id"] for r in read_jsonl(d / "spk.jsonl")}
        results = {r["id"]: r for r in read_jsonl(d / "results.jsonl")}
        for it in read_jsonl(out / "items.jsonl"):
            if it["stratum"] not in SPEAKER_STRATA or it["id"] in done or it["id"] not in results:
                continue
            if it["voice"]:
                pairs = [(out / "voices" / f"{it['voice']}.wav", d / f"{it['id']}.wav")]
            else:
                frags = [Path(w) for w in results[it["id"]]["wavs"][1:]]
                pairs = [(frags[0], f) for f in frags[1:]]
            sims = []
            for a, b in pairs:
                ea, eb = emb(a), emb(b)
                sims.append(None if ea is None or eb is None else float(ea @ eb))
            sim = None if not sims or any(s is None for s in sims) else min(sims)
            with open(d / "spk.jsonl", "a") as f:
                f.write(json.dumps(dict(id=it["id"], sim=sim, sims=sims)) + "\n")
        print(arm, "done", flush=True)


def fisher_greater(k1, n1, k0, n0):
    """One-sided Fisher exact p: P(at least k1 of the k1 + k0 flagged runs fall in the n1 group)."""
    total, flagged = n1 + n0, k1 + k0
    denom = math.comb(total, flagged)
    return sum(math.comb(n1, k) * math.comb(n0, flagged - k) for k in range(k1, min(n1, flagged) + 1)) / denom


def sign_flip_p(diffs, flips=20000):
    """One-sided paired permutation p for mean(diffs) > 0, signs flipped at random (seed 0)."""
    if not diffs:
        return 1.0
    rng = random.Random(0)
    observed = sum(diffs)
    hits = sum(1 for _ in range(flips) if sum(d if rng.random() < 0.5 else -d for d in diffs) >= observed)
    return (hits + 1) / (flips + 1)


def fences(values):
    """Tukey's far-out fences: quartiles -+ 3 interquartile ranges."""
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)


def load_runs(out):
    items = read_jsonl(out / "items.jsonl")
    runs = {}
    for arm in ARMS:
        res = {r["id"]: r for r in read_jsonl(out / arm / "results.jsonl")}
        asr = {r["id"]: r for r in read_jsonl(out / arm / "asr.jsonl")}
        spk = {r["id"]: r for r in read_jsonl(out / arm / "spk.jsonl")}
        for it in items:
            r = res.get(it["id"])
            if r is None or it["id"] not in asr or (it["stratum"] in SPEAKER_STRATA and it["id"] not in spk):
                sys.exit(f"arm {arm} item {it['id']} is not run or not scored")
            frames = sum(f["frames"] for f in r["fragments"])
            runs[arm, it["id"]] = dict(stopped=all(f["stopped"] for f in r["fragments"]), frames=frames, fpu=frames / it["units"],
                                       err=asr[it["id"]]["err"], sim=spk[it["id"]]["sim"] if it["id"] in spk else None, transcript=asr[it["id"]]["transcript"])
    return items, runs


def cmd_gate(args):
    out = Path(args.out).resolve()
    items, runs = load_runs(out)
    group = lambda it: (it["stratum"], it["lang"])
    groups = sorted({group(it) for it in items})
    strata = sorted({it["stratum"] for it in items})
    len_fence, sim_fence = {}, {}
    for g in groups:
        len_fence[g] = fences([runs[a, it["id"]]["fpu"] for it in items if group(it) == g for a in BF16])
    for s in SPEAKER_STRATA:
        sims = [runs[a, it["id"]]["sim"] for it in items if it["stratum"] == s for a in BF16]
        sim_fence[s] = fences([x for x in sims if x is not None])[0]
    by_id = {it["id"]: it for it in items}
    for (arm, id_), r in runs.items():
        it = by_id[id_]
        lo, hi = len_fence[group(it)]
        r["flags"] = dict(stop=not r["stopped"], wer=r["err"] > 0.20, len=not (lo <= r["fpu"] <= hi),
                          spk=it["stratum"] in SPEAKER_STRATA and (r["sim"] is None or r["sim"] < sim_fence[it["stratum"]]))
        r["any"] = any(r["flags"].values())

    lines, failed = [], []

    def say(s=""):
        lines.append(s)
        print(s)

    def count(arms, key, stratum=None):
        sel = [runs[a, it["id"]] for it in items if stratum in (None, it["stratum"]) for a in arms]
        return sum(1 for r in sel if (r["any"] if key == "any" else r["flags"][key])), len(sel)

    say(f"study {out}: {len(items)} items, {len(items) * 2} bf16 runs (A, C) and {len(items) * 2} int8 runs (B kernel, R reference)")
    say("fences from the bf16 arms: frames per unit " + ", ".join(f"{g[0]}/{g[1]} [{lo:.2f}, {hi:.2f}]" for g, (lo, hi) in len_fence.items())
        + "; speaker similarity floor " + ", ".join(f"{s} {v:.3f}" for s, v in sim_fence.items()))
    say()
    say("G1-G4 flagged runs, pooled over strata (fail: one-sided Fisher p < 0.05)")
    say("flag\tbf16 A\tbf16 C\tint8 B\tint8 R\tbf16\tint8\tp\tverdict")
    for key, name in (("stop", "G1 no end of speech"), ("wer", "G2 error rate > 0.20"), ("len", "G3 frames per unit outside fences"), ("spk", "G4 speaker similarity below floor")):
        per = {a: count((a,), key)[0] for a in ARMS}
        (k0, n0), (k1, n1) = count(BF16, key), count(INT8, key)
        if key == "spk":
            n0 = n1 = 2 * sum(1 for it in items if it["stratum"] in SPEAKER_STRATA)
        p = fisher_greater(k1, n1, k0, n0)
        ok = p >= 0.05
        if not ok:
            failed.append((name, p))
        say(f"{name}\t{per['A']}\t{per['C']}\t{per['B']}\t{per['R']}\t{k0}/{n0}\t{k1}/{n1}\t{p:.3f}\t{'pass' if ok else 'FAIL'}")
    say()
    say("G5-G6 paired means over (item, seed): A with B, C with R (fail: degradation beyond the margin, or sign-flip permutation p < 0.05)")
    say("metric\tpairs\tbf16 mean\tint8 mean\tint8 - bf16\tmargin\tp\tverdict")
    for key, name, margin, sign in (("err", "G5 error rate", 0.010, 1), ("sim", "G6 speaker similarity", 0.010, -1)):
        pairs = [(runs[a, it["id"]][key], runs[b, it["id"]][key]) for it in items for a, b in PAIRS]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        d = [sign * (y - x) for x, y in pairs]
        mean0, mean1 = statistics.fmean(x for x, _ in pairs), statistics.fmean(y for _, y in pairs)
        p = sign_flip_p(d)
        ok = statistics.fmean(d) <= margin and p >= 0.05
        if not ok:
            failed.append((name, p))
        say(f"{name}\t{len(pairs)}\t{mean0:.4f}\t{mean1:.4f}\t{mean1 - mean0:+.4f}\t{'+' if sign > 0 else '-'}{margin}\t{p:.3f}\t{'pass' if ok else 'FAIL'}")
    say()
    say("G7 per stratum, runs with any flag (fail: one-sided Fisher p < 0.02); error rate and frames per unit for the reader")
    say("stratum\truns per side\tbf16 flagged\tint8 flagged\tp\tverdict\tbf16 err\tint8 err\tbf16 frames/unit\tint8 frames/unit\tbf16 sim\tint8 sim")
    for s in strata:
        (k0, n0), (k1, n1) = count(BF16, "any", s), count(INT8, "any", s)
        p = fisher_greater(k1, n1, k0, n0)
        ok = p >= 0.02
        if not ok:
            failed.append((f"G7 {s}", p))
        mean = lambda arms, key: statistics.fmean([runs[a, it["id"]][key] for it in items if it["stratum"] == s for a in arms if runs[a, it["id"]][key] is not None] or [float("nan")])
        say(f"{s}\t{n0}\t{k0}\t{k1}\t{p:.3f}\t{'pass' if ok else 'FAIL'}\t{mean(BF16, 'err'):.4f}\t{mean(INT8, 'err'):.4f}\t{mean(BF16, 'fpu'):.2f}\t{mean(INT8, 'fpu'):.2f}\t{mean(BF16, 'sim'):.3f}\t{mean(INT8, 'sim'):.3f}")
    say()
    say("noise floor (not part of the gate): how two arms differ on the same items")
    say("pair\twhat differs\tflag discordant\tmean |err diff|\tmedian |log frames ratio|\tidentical frame counts")
    for a, b, what in (("A", "C", "seed"), ("B", "R", "seed (and kernel against reference weights)"), ("A", "B", "format, same seed"), ("C", "R", "format, same seed")):
        ra, rb = [runs[a, it["id"]] for it in items], [runs[b, it["id"]] for it in items]
        disc = sum(1 for x, y in zip(ra, rb) if x["any"] != y["any"])
        err = statistics.fmean(abs(x["err"] - y["err"]) for x, y in zip(ra, rb))
        ratio = statistics.median(abs(math.log(max(x["frames"], 1) / max(y["frames"], 1))) for x, y in zip(ra, rb))
        same = sum(1 for x, y in zip(ra, rb) if x["frames"] == y["frames"])
        say(f"{a} vs {b}\t{what}\t{disc}\t{err:.4f}\t{ratio:.3f}\t{same}/{len(ra)}")
    say()
    say("flagged runs")
    for (arm, id_), r in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if r["any"]:
            flags = ",".join(k for k, v in r["flags"].items() if v)
            say(f"{arm}\t{id_}\t{flags}\tframes {r['frames']} ({r['fpu']:.2f}/unit)\terr {r['err']:.3f}\tsim {r['sim']}\t{r['transcript'][:100]}")
    say()
    verdict = "PASS" if not failed else "FAIL: " + "; ".join(f"{n} (p {p:.3f})" for n, p in failed)
    say(f"GATE {verdict}")
    (out / "gate.txt").write_text("\n".join(lines) + "\n")
    sys.exit(0 if not failed else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("plan", cmd_plan), ("run", cmd_run), ("asr", cmd_asr), ("spk", cmd_spk), ("gate", cmd_gate)):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
        s.add_argument("--chunk", type=int, default=20)
        s.add_argument("--voices", action="store_true", help="asr: also transcribe the reference voices")
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
