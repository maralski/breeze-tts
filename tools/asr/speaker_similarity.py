#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Cosine similarity of CAM++ speaker embeddings between WAVs: do they sound like one speaker?

    tools/asr/speaker_similarity.py [--onnx campplus.onnx] WAV WAV [WAV ...]

--onnx defaults to CAMPPLUS_ONNX.

Prints every pair's cosine. CAM++ (3D-Speaker, as CosyVoice and Ming-3.0-tts ship it) embeds
80-bin Kaldi fbank features of 16 kHz audio, mean-normalized over time. Needs onnxruntime and
torchaudio, and campplus.onnx, which the Ming-3.0-tts checkpoint ships; neither comes with
Breeze TTS 2.
"""
import argparse, itertools, os, subprocess

import numpy as np
import onnxruntime
import torch
import torchaudio.compliance.kaldi as kaldi


def read(path):
    """Mono float samples at 16 kHz, decoded by ffmpeg (float WAVs included)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy())


def embed(session, path):
    feat = kaldi.fbank(read(path).unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    name = session.get_inputs()[0].name
    e = session.run(None, {name: feat.unsqueeze(0).numpy()})[0].flatten()
    return e / np.linalg.norm(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default=os.environ.get("CAMPPLUS_ONNX"), required="CAMPPLUS_ONNX" not in os.environ)
    p.add_argument("wavs", nargs="+")
    args = p.parse_args()
    session = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    emb = {w: embed(session, w) for w in args.wavs}
    for a, b in itertools.combinations(args.wavs, 2):
        print(f"{float(emb[a] @ emb[b]):.3f}  {os.path.basename(a)}  {os.path.basename(b)}")


if __name__ == "__main__":
    main()
