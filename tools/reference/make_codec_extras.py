# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Codec goldens that the five generation cases do not reach.

Every generation case is shorter than the decoder transformer's 72-frame window, and the
reference clip is a whole number of frames. This writes, under <goldens>/codec_extra/:

  long/      the design_en greedy codes tiled to 150 frames, through one full decoder forward
             (same tensors as <case>/codec/, without the stream runs)
  enc_odd/   the first 50001 samples of ref_clip_en.wav through the encoder: a partial last
             frame and an odd 25 Hz length, so the right pads run
  enc_long/  the first 287001 samples of long/pcm_full (12 s, 299 frames at 25 Hz, past the 250 the
             config names as an attention window), as PCM16, through the encoder

Environment: the repo venv, CPU only. BREEZE_WEIGHTS is the checkpoint dir; BREEZE_GOLDENS
overrides <repo>/.audit/goldens. Run make_goldens.py first.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSTokenizer

WEIGHTS = Path(os.environ["BREEZE_WEIGHTS"]).resolve()
GOLDENS = Path(os.environ.get("BREEZE_GOLDENS", Path(__file__).resolve().parents[2] / ".audit" / "goldens"))
OUT = GOLDENS / "codec_extra"
LONG_FRAMES = 150


def put(directory: Path, name: str, tensor) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if np.issubdtype(array.dtype, np.integer):
        array.astype("<i4").tofile(directory / f"{name}.i32")
    else:
        array.astype("<f4").tofile(directory / f"{name}.f32")
    print(directory.name, name, list(array.shape), flush=True)


@torch.inference_mode()
def decoder_stages(decoder, codes: torch.Tensor) -> dict[str, torch.Tensor]:
    x = codes.T[None].long()
    stages = {}
    h = decoder.quantizer.decode(x)
    stages["quantizer"] = h[0]
    h = decoder.pre_conv(h)
    stages["pre_conv"] = h[0]
    h = decoder.pre_transformer(inputs_embeds=h.transpose(1, 2)).last_hidden_state
    stages["pre_transformer"] = h[0]
    h = h.permute(0, 2, 1)
    for i, blocks in enumerate(decoder.upsample):
        for block in blocks:
            h = block(h)
        stages[f"upsample{i}"] = h[0]
    for i, block in enumerate(decoder.decoder):
        h = block(h)
        stages[f"decoder{i}"] = h[0]
    stages["pcm_full"] = h.clamp(min=-1, max=1)[0, 0]
    assert torch.equal(decoder(x)[0, 0], stages["pcm_full"])
    return stages


@torch.inference_mode()
def encoder_stages(atok, wav: np.ndarray, directory: Path) -> None:
    enc = atok.model.encoder
    stores, handles = {}, []
    for name, module in (("seanet", enc.encoder), ("transformer", enc.encoder_transformer), ("downsample", enc.downsample)):
        store: list[torch.Tensor] = []
        stores[name] = store
        handles.append(module.register_forward_hook(
            lambda _m, _i, out, store=store: store.append((out if isinstance(out, torch.Tensor) else out[0]).detach().float().clone())))
    try:
        codes = torch.as_tensor(atok.encode(wav, sr=24000)["audio_codes"][0]).long()
    finally:
        for h in handles:
            h.remove()
    put(directory, "input_samples", wav)
    for name, store in stores.items():
        assert len(store) == 1, (name, len(store))
        put(directory, name, store[0][0])
    put(directory, "codes", codes)


def main() -> None:
    torch.set_grad_enabled(False)
    atok = Qwen3TTSTokenizer.from_pretrained(str(WEIGHTS / "audio_tokenizer"), device_map="cpu")
    base = np.fromfile(GOLDENS / "design_en" / "codec" / "codes.i32", dtype="<i4").reshape(-1, 16)
    codes = torch.from_numpy(np.tile(base, ((LONG_FRAMES + len(base) - 1) // len(base), 1))[:LONG_FRAMES].copy())
    stages = decoder_stages(atok.model.decoder, codes)
    put(OUT / "long", "codes", codes)
    for name, value in stages.items():
        put(OUT / "long", name, value)

    clip, sr = sf.read(GOLDENS / "ref_clip_en.wav", dtype="float32")
    assert sr == 24000
    encoder_stages(atok, clip[:50001], OUT / "enc_odd")
    pcm16 = np.round(np.clip(stages["pcm_full"].numpy()[:287001], -1, 1) * 32767).astype(np.int16)
    encoder_stages(atok, pcm16.astype(np.float32) / 32768, OUT / "enc_long")


if __name__ == "__main__":
    main()
