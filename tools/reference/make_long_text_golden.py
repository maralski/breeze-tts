# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Text-encoder goldens for one segment long enough (> 256 tokens) to engage the sliding window.

The five main cases have segments of at most 34 tokens, where the window is a no-op. This writes
`<goldens>/long_text/` in the main layout, with only the text-encoder tensors of one branch:

  BREEZE_WEIGHTS=... BREEZE_REF_SRC=... .venv/bin/python tools/reference/make_long_text_golden.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

REF_SRC = Path(os.environ["BREEZE_REF_SRC"]).resolve()
WEIGHTS = Path(os.environ["BREEZE_WEIGHTS"]).resolve()
OUT = Path(os.environ.get("BREEZE_GOLDENS", Path(__file__).resolve().parents[2] / ".audit" / "goldens")) / "long_text"
sys.path.insert(0, str(REF_SRC))
from breeze_infer.runtime import load_runtime  # noqa: E402

TEXT = (
    "[S0]The harbour was quiet before dawn, and the fishing boats rocked gently against the pier while gulls "
    "circled overhead. Maria walked along the stones with a lantern, counting the ropes and checking every knot "
    "the way her grandfather had taught her, slowly, patiently, never skipping one. "
) * 6
LAYERS = [0, 4, 5, 25]


@torch.inference_mode()
def main() -> None:
    tok, model, _ = load_runtime(WEIGHTS, device="cpu", attn_implementation="eager")
    model.float()
    te = model.text_encoder
    ids = torch.tensor(tok(TEXT, add_special_tokens=True)["input_ids"])
    n = ids.shape[0]
    assert n > 300, n
    stores: dict[str, torch.Tensor] = {}
    handles = [te.embed_tokens.register_forward_hook(lambda _m, _i, o: stores.__setitem__("embed", o[0]))]
    for li in LAYERS:
        handles.append(te.layers[li].register_forward_hook(
            lambda _m, _i, o, li=li: stores.__setitem__(f"layer{li}", (o[0] if isinstance(o, tuple) else o)[0])))
    hidden = te(input_ids=ids[None], attention_mask=torch.ones(1, n, dtype=torch.long), position_ids=torch.arange(n)[None]).last_hidden_state
    for h in handles:
        h.remove()
    stores["final_norm"] = hidden[0]
    stores["proj"] = model.text_encoder_proj(hidden)[0]
    (OUT / "fp32").mkdir(parents=True, exist_ok=True)
    tensors = {}
    for key, t in stores.items():
        name = f"text_cond_seg0_{key}"
        t.float().contiguous().numpy().tofile(OUT / "fp32" / f"{name}.f32")
        tensors[name] = {"file": f"fp32/{name}.f32", "shape": list(t.shape), "dtype": "float32"}
    manifest = {
        "case": "long_text",
        "cfg_scale": 1.0,
        "tokens": {"cond": {"input_ids": ids.tolist(), "segments": [ids.tolist()]}},
        "runs": {"fp32": {"tensors": tensors}},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {OUT} ({n} tokens)")


if __name__ == "__main__":
    main()
