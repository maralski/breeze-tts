# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Check spec claims against the checkpoint headers and the instantiated model.

Environment: the repo venv (`.venv`, torch 2.9.1, transformers 4.57.3,
qwen-tts 0.1.1), CPU only. Env vars BREEZE_WEIGHTS (checkpoint dir) and
BREEZE_REF_SRC (reference clone). Prints JSON to stdout.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

WEIGHTS = Path(os.environ["BREEZE_WEIGHTS"]).resolve()
sys.path.insert(0, str(Path(os.environ["BREEZE_REF_SRC"]).resolve()))

from breeze_infer.runtime import load_runtime  # noqa: E402


def headers() -> dict[str, dict]:
    keys = {}
    for path in sorted(WEIGHTS.glob("model-*.safetensors")):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        keys.update(header)
    return keys


def llama3_inv_freq() -> np.ndarray:
    inv = 1.0 / (500000.0 ** (np.arange(0, 128, 2, dtype=np.float32) / 128)).astype(np.float32)
    wl = 2 * math.pi / inv
    lw, hw = 16 / 0.001953125, 16 / 0.0078125
    inv1 = np.where(wl > lw, inv / 32, inv)
    s = (16 / wl - 0.001953125) / (0.0078125 - 0.001953125)
    smooth = (1 - s) * inv1 / 32 + s * inv1
    return np.where((wl >= hw) & (wl <= lw), smooth, inv1).astype(np.float32)


def main() -> None:
    keys = headers()
    report: dict = {}
    bb_qn = [k for k in keys if k.startswith("backbone_model.layers.") and ".self_attn.q_norm." in k]
    bb_kn = [k for k in keys if k.startswith("backbone_model.layers.") and ".self_attn.k_norm." in k]
    dd_norms = [k for k in keys if k.startswith("depth_decoder.") and ("q_norm" in k or "k_norm" in k)]
    report["backbone_q_norm_keys"] = len(bb_qn)
    report["backbone_k_norm_keys"] = len(bb_kn)
    report["backbone_q_norm_shape"] = keys[bb_qn[0]]["shape"] if bb_qn else None
    report["depth_qk_norm_keys"] = dd_norms
    report["codebooks_head"] = keys.get("depth_decoder.codebooks_head.weight", {}).get("shape")
    report["backbone_embed_keys"] = [k for k in keys if k.startswith("backbone_model.embed")]
    report["depth_embed_tokens"] = keys.get("depth_decoder.model.embed_tokens.weight", {}).get("shape")
    report["all_embed_like_keys"] = sorted(k for k in keys if "embed" in k and not k.startswith("codec_model."))

    _, model, _ = load_runtime(WEIGHTS, device="cpu", attn_implementation="eager")
    bb, dd, te = model.backbone_model, model.depth_decoder.model, model.text_encoder
    report["tied_embedding_same_storage"] = (
        bb.embed_tokens.embed_audio_tokens.weight.data_ptr() == dd.embed_tokens.weight.data_ptr()
    )

    def eps(norm):
        return getattr(norm, "variance_epsilon", getattr(norm, "eps", None))

    bb_inv = bb.rotary_emb.inv_freq.float().numpy()
    want_bb = 1.0 / (1e6 ** (np.arange(0, 128, 2, dtype=np.float32) / 128))
    report["backbone"] = {
        "input_layernorm_eps": eps(bb.layers[0].input_layernorm),
        "q_norm_eps": eps(bb.layers[0].self_attn.q_norm),
        "final_norm_eps": eps(bb.norm),
        "rope_type": getattr(bb.rotary_emb, "rope_type", None),
        "attention_scaling": float(getattr(bb.rotary_emb, "attention_scaling", 1.0)),
        "inv_freq_len": len(bb_inv),
        "inv_freq_dtype": str(bb.rotary_emb.inv_freq.dtype),
        "inv_freq_max_rel_err_vs_theta_1e6": float(np.max(np.abs(bb_inv - want_bb) / want_bb)),
        "attn_scaling_factor": float(bb.layers[0].self_attn.scaling),
        "sliding_window": getattr(bb.layers[0].self_attn, "sliding_window", None),
    }
    dd_inv = dd.rotary_emb.inv_freq.float().numpy()
    want_dd = llama3_inv_freq()
    report["depth"] = {
        "input_layernorm_eps": eps(dd.layers[0].input_layernorm),
        "final_norm_eps": eps(dd.norm),
        "rope_type": getattr(dd.rotary_emb, "rope_type", None),
        "attention_scaling": float(getattr(dd.rotary_emb, "attention_scaling", 1.0)),
        "inv_freq_dtype": str(dd.rotary_emb.inv_freq.dtype),
        "inv_freq_max_rel_err_vs_spec_llama3": float(np.max(np.abs(dd_inv - want_dd) / want_dd)),
        "inv_freq_29_32_34": [float(dd_inv[29]), float(dd_inv[32]), float(dd_inv[34])],
        "has_q_norm": hasattr(dd.layers[0].self_attn, "q_norm"),
        "attn_scaling_factor": float(dd.layers[0].self_attn.scaling),
    }
    te_bufs = {
        name: [len(b), float(b.float()[0]), float(b.float()[-1]), str(b.dtype)]
        for name, b in te.named_buffers()
        if "inv_freq" in name and "original" not in name
    }
    report["text_encoder"] = {
        "inv_freq_buffers[len, first, last, dtype]": te_bufs,
        "embed_scale": [float(te.embed_tokens.embed_scale), str(te.embed_tokens.embed_scale.dtype)],
        "norm_eps": eps(te.norm),
        "layer_types": list(getattr(te.config, "layer_types", [])),
    }
    report["bf16_float_buffers"] = sorted(
        name for name, b in model.named_buffers() if b.dtype == torch.bfloat16 and not name.startswith("codec_model.")
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
