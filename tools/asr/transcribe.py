#!/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Transcribe WAVs with Whisper and score them against the text they should say.

    tools/asr/transcribe.py [--lang en|zh] [--max-error 0.1] WAV TEXT [WAV TEXT ...]

Needs openai-whisper and torch (the system Python 3.14 has both; the project's .venv has
neither) and ffmpeg. It reads the MLX Whisper weights already in the Hugging Face cache instead
of downloading a checkpoint: mlx-community/whisper-small.en-mlx for English and
mlx-community/whisper-large-v3-turbo for Chinese. MLX stores conv kernels as [out, k, in];
torch wants [out, in, k], and the MLP layers are named mlp1 and mlp2
for torch's mlp.0 and mlp.2.

English is scored by word error rate after lowercasing and dropping punctuation; Chinese by
character error rate over Han characters, letters and digits. Vocal-event tags such as
"(laugh)" and "[笑]" are removed from the expected text first. Prints one block per file and
exits 1 when any error rate exceeds --max-error.
"""
import argparse, glob, json, os, re, sys

import numpy as np
import torch
import whisper
from whisper.model import ModelDimensions, Whisper

MODELS = {"en": "mlx-community--whisper-small.en-mlx", "zh": "mlx-community--whisper-large-v3-turbo"}


def read_safetensors(path):
    """The tensors of a .safetensors file; F16, BF16, F32 and I64 only."""
    raw = open(path, "rb").read()
    n = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + n])
    body = memoryview(raw)[8 + n:]
    out = {}
    for name, e in header.items():
        if name == "__metadata__":
            continue
        a, b = e["data_offsets"]
        if e["dtype"] == "BF16":
            bits = np.frombuffer(body[a:b], dtype="<u2").astype(np.uint32) << 16
            value = bits.view(np.float32)
        else:
            value = np.frombuffer(body[a:b], dtype={"F16": "<f2", "F32": "<f4", "I64": "<i8"}[e["dtype"]])
        out[name] = value.reshape(e["shape"])
    return out


def load(lang):
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snap = sorted(glob.glob(os.path.join(hub, "models--" + MODELS[lang], "snapshots", "*")))[-1]
    config = json.load(open(os.path.join(snap, "config.json")))
    dims = ModelDimensions(**{k: v for k, v in config.items() if k != "model_type"})
    path = os.path.join(snap, "weights.npz")
    if os.path.exists(path):
        weights = dict(np.load(path))
    else:
        weights = read_safetensors(os.path.join(snap, "weights.safetensors"))
    state = {}
    for name, value in weights.items():
        if name == "alignment_heads":
            continue
        name = name.replace(".mlp1.", ".mlp.0.").replace(".mlp2.", ".mlp.2.")
        t = torch.from_numpy(np.asarray(value, dtype=np.float32))
        if name.endswith("conv1.weight") or name.endswith("conv2.weight"):
            t = t.permute(0, 2, 1).contiguous()
        state[name] = t
    model = Whisper(dims)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if m != "encoder.positional_embedding"]
    if missing or unexpected:
        sys.exit(f"weights do not fit the model: missing {missing}, unexpected {unexpected}")
    return model.eval()


TAGS = re.compile(r"\([^()]{1,32}\)|\[[^\[\]]{1,32}\]")


def tokens(text, lang):
    text = TAGS.sub(" ", text).lower()
    if lang == "zh":
        return [c for c in text if c.isalnum()]
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.replace("’", "'"))


def error_rate(want, got):
    """Levenshtein distance over tokens, divided by the expected count."""
    d = list(range(len(got) + 1))
    for i, w in enumerate(want, 1):
        prev, d[0] = d[0], i
        for j, g in enumerate(got, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (w != g))
    return d[len(got)] / max(1, len(want))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=MODELS, default="en")
    p.add_argument("--max-error", type=float, default=0.1)
    p.add_argument("pairs", nargs="+")
    args = p.parse_args()
    if len(args.pairs) % 2:
        sys.exit("give WAV TEXT pairs")
    model = load(args.lang)
    options = dict(language=args.lang, fp16=False, temperature=0.0, condition_on_previous_text=False)
    if args.lang == "zh":
        options["initial_prompt"] = "以下是普通话的句子。"
    failed = False
    for wav, want in zip(args.pairs[::2], args.pairs[1::2]):
        got = whisper.transcribe(model, wav, **options)["text"].strip()
        rate = error_rate(tokens(want, args.lang), tokens(got, args.lang))
        failed |= rate > args.max_error
        unit = "cer" if args.lang == "zh" else "wer"
        print(f"file {wav}\n  expected   {want}\n  transcript {got}\n  {unit} {rate:.3f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
