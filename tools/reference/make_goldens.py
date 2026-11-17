# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Capture golden tensors from the upstream Breeze TTS 2 reference on CPU.

Environment: the repo venv (`.venv`, torch 2.9.1, transformers 4.57.3,
qwen-tts 0.1.1, numpy, soundfile), CPU only. Required env vars:
  BREEZE_WEIGHTS  checkpoint dir (the HF snapshot of BreezeBlue/Breeze-TTS-2)
  BREEZE_REF_SRC  clone of github.com/breezeblue-ai/breeze-tts (read only)
Optional: BREEZE_GOLDENS (default <repo>/.audit/goldens).

Run the fp32 pass first, then the bf16 pass (it replays the fp32 codes):
  python tools/reference/make_goldens.py --precision fp32
  python tools/reference/make_goldens.py --precision bf16
See tools/reference/README.md for the output format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REF_SRC = Path(os.environ["BREEZE_REF_SRC"]).resolve()
WEIGHTS = Path(os.environ["BREEZE_WEIGHTS"]).resolve()
GOLDENS = Path(
    os.environ.get(
        "BREEZE_GOLDENS", Path(__file__).resolve().parents[2] / ".audit" / "goldens"
    )
)
sys.path.insert(0, str(REF_SRC))

import models.fast_streaming as FS  # noqa: E402
from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze  # noqa: E402
from breeze_infer.templates import get_template, prepare_inputs  # noqa: E402
from models.cudagraph.depth_decoder_graph import DepthDecoderGraph  # noqa: E402
from models.stream_runtime import MultiRequestStreamRuntime, QwenStreamRuntimeConfig  # noqa: E402
from models.stream_runtime.stream import kv_cache as KV  # noqa: E402

ORIGINAL_KV_UPDATE = KV.StaticShiftKVCache.update

SEED = 42
FRAME_CAP = 120
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920
REF_CLIP_FRAMES = 50  # 4.0 s
REF_CLIP = GOLDENS / "ref_clip_en.wav"
TEXT_LAYERS = (0, 4, 5, 25)
BACKBONE_LAYERS = (0, 27)
EOS = 2051

DESIGN_EN_TEXT = "Welcome aboard. Your journey begins now."
CASES: dict[str, dict] = {
    "design_en": {
        "template": "tts_instruction",
        "text": DESIGN_EN_TEXT,
        "instruction": "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery.",
        "cfg_scale": 4.0,
    },
    "plain_en": {
        "template": "tts_plain",
        "text": "Thank you for calling. How can I help you today?",
        "cfg_scale": 1.0,
    },
    "design_zh": {
        "template": "tts_instruction",
        "text": "[笑] 欢迎回来，今天过得怎么样？",
        "instruction": "一位活泼开朗的年轻男性，声音明亮，语速适中。",
        "cfg_scale": 4.0,
    },
    "clone_en": {
        "template": "ref_clone_tata",
        "text": "Please keep your seatbelt fastened until the sign turns off.",
        "ref": True,
        "cfg_scale": 1.0,
    },
    "direction_en": {
        "template": "ref_edit_tata",
        "text": "Please keep your seatbelt fastened until the sign turns off.",
        "instruction": "Speak slowly and softly, almost in a whisper.",
        "ref": True,
        "cfg_scale": 4.0,
    },
}


# ---------------------------------------------------------------- output format


class TensorDir:
    """Writes raw little-endian tensors under <case>/<sub>/ and indexes them."""

    def __init__(self, case_dir: Path, sub: str):
        self.case_dir = case_dir
        self.sub = sub
        (case_dir / sub).mkdir(parents=True, exist_ok=True)
        self.index: dict[str, dict] = {}

    def put(self, name: str, value, dtype: str = "float32") -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            value = value.float().numpy() if dtype == "float32" else value.long().numpy()
        arr = np.ascontiguousarray(np.asarray(value), dtype="<f4" if dtype == "float32" else "<i4")
        ext = "f32" if dtype == "float32" else "i32"
        rel = f"{self.sub}/{name}.{ext}"
        arr.tofile(self.case_dir / rel)
        self.index[name] = {"file": rel, "shape": list(arr.shape), "dtype": dtype}

    def i32(self, name: str, value) -> None:
        self.put(name, value, "int32")


def update_manifest(case_dir: Path, **sections) -> None:
    path = case_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict) and key == "runs":
            manifest[key].update(value)
        else:
            manifest[key] = value
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")


def write_wav(path: Path, audio: np.ndarray) -> None:
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")


# ---------------------------------------------------------------- reference plumbing


class CpuStreamingRuntime(FS.FastBreezeStreamingRuntime):
    """The reference runtime minus its CUDA-only guard (F:202-203).

    The guard is the last statement of __init__ after every attribute is set,
    so swallowing exactly that error leaves an identical object.
    """

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except RuntimeError as exc:
            if str(exc) != "fast streaming requires a CUDA device":
                raise
            if self.config.repetition_penalty <= 0:
                raise ValueError("repetition_penalty must be > 0") from exc


def fix_codec_kv_shift() -> None:
    """Make StaticShiftKVCache.update (SK:69-73) deterministic on CPU.

    The reference shifts the window with an overlapping in-place
    `k[..., :-s, :].copy_(k[..., s:, :])`. With several CPU threads that copy
    races, and repeated decodes of the same codes differ by up to ~1.5e-3 from
    frame 24 on. This version computes the same shift from a copy of the
    source, which is what the reference code intends.
    """
    original = ORIGINAL_KV_UPDATE

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        layer = self.layers[layer_idx]
        shift = min(int(key_states.shape[-2]), self.window)
        if shift >= self.window:
            return original(self, key_states, value_states, layer_idx, cache_kwargs)
        layer.k.copy_(torch.cat([layer.k[..., shift:, :].clone(), key_states[..., -shift:, :]], dim=-2))
        layer.v.copy_(torch.cat([layer.v[..., shift:, :].clone(), value_states[..., -shift:, :]], dim=-2))
        return layer.k, layer.v

    KV.StaticShiftKVCache.update = update


def hook_outputs(module, pick=lambda out: out):
    store: list[torch.Tensor] = []

    def fn(_mod, _inp, out):
        store.append(pick(out).detach().float().clone())

    return store, module.register_forward_hook(fn)


def first(out):
    return out if isinstance(out, torch.Tensor) else out[0]


def inverse_cdf(probs: torch.Tensor, u: float) -> int:
    """Token = first index i (natural vocab order) with cdf[i] > u * cdf[-1].

    cdf is the float64 cumulative sum of the float32 probability row. The
    comparison is strict, so zero-probability entries are never chosen.
    """
    p = probs.detach().double().cpu().numpy()
    cdf = np.cumsum(p)
    hit = np.nonzero(cdf > float(u) * cdf[-1])[0]
    if len(hit):
        return int(hit[0])
    return int(np.nonzero(p > 0)[0][-1])


def make_uniforms(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    u_backbone = rng.random(FRAME_CAP + 1).astype(np.float32)
    u_depth = rng.random((FRAME_CAP, 15)).astype(np.float32)
    return u_backbone, u_depth


class StreamRecorder:
    """Runs FastBreezeStreamingRuntime.iter_audio_chunks and records every step.

    mode "greedy": do_sample False on backbone and depth.
    mode "sampled": torch.multinomial is replaced by inverse_cdf over recorded
    uniforms (backbone draw m uses u_backbone[m]; depth draw for frame f,
    codebook c+1 uses u_depth[f, c]).
    forced: optional (codes [N,16], stopped_on_eos) that overrides every token
    after the reference sampler ran, so a second precision follows the
    canonical stream; the sampler's own choices are still recorded.
    """

    def __init__(self, mode: str, forced: tuple[np.ndarray, bool] | None = None):
        self.mode = mode
        self.forced = forced
        self.u_backbone, self.u_depth = make_uniforms(SEED)
        self.bb_cfg: list[torch.Tensor] = []
        self.bb_own: list[int] = []
        self.bb_used: list[int] = []
        self.depth_hidden: list[torch.Tensor] = []
        self.depth_cfg: list[torch.Tensor] = []
        self.depth_own: list[int] = []
        self.frames: list[torch.Tensor] = []
        self._cb = 0
        self._site: tuple = ()

    def _forced_backbone(self, step: int) -> int | None:
        if self.forced is None:
            return None
        codes, eos = self.forced
        if step < len(codes):
            return int(codes[step, 0])
        assert eos and step == len(codes), "forced replay ran past the canonical stream"
        return EOS

    def _multinomial(self, probs, num_samples, *args, **kwargs):
        assert num_samples == 1 and probs.shape[0] == 1, probs.shape
        if self._site[0] == "backbone":
            assert probs.shape[-1] == 2052
            u = self.u_backbone[self._site[1]]
        else:
            assert probs.shape[-1] == 2051
            u = self.u_depth[self._site[1], self._site[2]]
        return torch.tensor([[inverse_cdf(probs[0], u)]], dtype=torch.long)

    @contextmanager
    def patched(self, model):
        orig_sample = FS.sample_logits
        orig_cfg_sample = DepthDecoderGraph._cfg_sample
        orig_run = DepthDecoderGraph.run
        orig_multinomial = torch.multinomial
        rec = self

        def sample_logits(logits, **kwargs):
            step = len(rec.bb_cfg)
            rec.bb_cfg.append(logits.detach().float().clone()[0])
            rec._site = ("backbone", step)
            token = orig_sample(logits, **kwargs)
            rec.bb_own.append(int(token.item()))
            forced = rec._forced_backbone(step)
            if forced is not None:
                token = torch.full_like(token, forced)
            rec.bb_used.append(int(token.item()))
            return token

        def run(ddg, backbone_hidden, first_cb_token, **kwargs):
            rec.depth_hidden.append(backbone_hidden.detach().float().clone())
            rec._cb = 0
            return orig_run(ddg, backbone_hidden, first_cb_token, **kwargs)

        def cfg_sample(ddg, logits):
            frame = len(rec.depth_hidden) - 1
            if ddg.batch_size >= 2:
                cond = logits[: ddg.half, 0, :]
                uncond = logits[ddg.half :, 0, :]
                cfg = uncond + ddg.guidance_scale * (cond - uncond)
            else:
                cfg = logits[: ddg.half, 0, :]
            rec.depth_cfg.append(cfg.detach().float().clone()[0])
            rec._site = ("depth", frame, rec._cb)
            orig_cfg_sample(ddg, logits)
            rec.depth_own.append(int(ddg._tok_buf[0].item()))
            if rec.forced is not None:
                ddg._tok_buf.fill_(int(rec.forced[0][frame, rec._cb + 1]))
            rec._cb += 1

        lm_store, h1 = hook_outputs(model.lm_head)
        head_store, h2 = hook_outputs(model.depth_decoder.codebooks_head)
        FS.sample_logits = sample_logits
        DepthDecoderGraph._cfg_sample = cfg_sample
        DepthDecoderGraph.run = run
        if self.mode == "sampled":
            torch.multinomial = self._multinomial
        try:
            yield lm_store, head_store
        finally:
            FS.sample_logits = orig_sample
            DepthDecoderGraph._cfg_sample = orig_cfg_sample
            DepthDecoderGraph.run = orig_run
            torch.multinomial = orig_multinomial
            h1.remove()
            h2.remove()

    def run(self, model, atok, tok, inputs) -> dict:
        config = FS.FastStreamingConfig(
            max_new_tokens=FRAME_CAP,
            max_seq_len=2048,
            repetition_penalty=1.1,
            do_sample=False if self.mode == "greedy" else None,
        )
        rt = CpuStreamingRuntime(model, atok, config, tokenizer=tok)
        audio = []
        started = time.time()
        with self.patched(model) as (lm_store, head_store):
            for chunk in rt.iter_audio_chunks(
                inputs, seed=SEED, token_observer=lambda f: self.frames.append(f.clone())
            ):
                audio.append(chunk.audio)
        wall = time.time() - started
        n = len(self.frames)
        m = len(self.bb_cfg)
        stopped_on_eos = self.bb_used[-1] == EOS
        assert m == n + (1 if stopped_on_eos else 0), (m, n)
        assert len(lm_store) == m and len(head_store) == 15 * n, (len(lm_store), len(head_store))
        branches = lm_store[0].shape[0]
        return {
            "wall_s": wall,
            "stopped_on_eos": stopped_on_eos,
            "n_frames": n,
            "branches": branches,
            "codes": torch.stack(self.frames).long(),
            "backbone_hidden": torch.stack(self.depth_hidden),  # [N,B,2048]
            "backbone_logits": torch.stack(lm_store),  # [M,B,2052]
            "backbone_logits_cfg": torch.stack(self.bb_cfg),  # [M,2052]
            "backbone_token_own": torch.tensor(self.bb_own),
            "depth_logits": torch.stack(head_store)[:, :, 0, :].reshape(n, 15, branches, 2051),
            "depth_logits_cfg": torch.stack(self.depth_cfg).reshape(n, 15, 2051),
            "depth_token_own": torch.tensor(self.depth_own).reshape(n, 15),
            "u_backbone": self.u_backbone[:m],
            "u_depth": self.u_depth[:n],
            "audio": np.concatenate(audio) if audio else np.zeros(0, np.float32),
        }


# ---------------------------------------------------------------- inputs


def build_request(case: dict) -> dict:
    request = {"id": "golden", "text": case["text"], "speaker": "S0"}
    if case.get("instruction"):
        request["instruction"] = case["instruction"]
    if case.get("ref"):
        clip = json.loads(REF_CLIP.with_suffix(".json").read_text())
        request["ref_audio_path"] = str(REF_CLIP)
        request["ref_text"] = clip["transcript"]
    return request


def branch_specs(inputs: dict) -> dict[str, dict]:
    specs = {
        "cond": {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "text_ids_mask": inputs["text_ids_mask"],
            "text_ids_len": inputs["text_ids_len"],
            "input_values": inputs.get("input_values"),
        }
    }
    if "cfg_negative_prompt_ids" in inputs:
        specs["uncond"] = {
            "input_ids": inputs["cfg_negative_prompt_ids"],
            "attention_mask": inputs["cfg_negative_prompt_attention_mask"],
            "text_ids_mask": inputs["cfg_negative_text_ids_mask"],
            "text_ids_len": inputs["cfg_negative_text_ids_len"],
            "input_values": inputs.get("cfg_negative_input_values"),
        }
    return specs


def segments_of(spec: dict) -> list[torch.Tensor]:
    ids = spec["input_ids"][0][spec["text_ids_mask"][0]]
    return list(torch.split(ids, spec["text_ids_len"].tolist()))


def token_manifest(tok, specs: dict) -> dict:
    out = {}
    for name, spec in specs.items():
        out[name] = {
            "input_ids": spec["input_ids"][0].tolist(),
            "text_ids_mask": [int(x) for x in spec["text_ids_mask"][0].tolist()],
            "text_ids_len": spec["text_ids_len"].tolist(),
            "segments": [s.tolist() for s in segments_of(spec)],
            "rendered": tok.decode(spec["input_ids"][0], skip_special_tokens=False),
        }
    return out


# ---------------------------------------------------------------- module goldens


@torch.inference_mode()
def text_encoder_goldens(model, specs: dict, out: TensorDir) -> dict:
    te = model.text_encoder
    notes = {}
    for branch, spec in specs.items():
        merged = model._merge_input_ids_with_input_values(**spec)["inputs_embeds"][0]
        text_rows = merged[spec["text_ids_mask"][0]].float()
        proj_rows = []
        for si, seg in enumerate(segments_of(spec)):
            stores, handles = {}, []
            stores["embed"], h = hook_outputs(te.embed_tokens)
            handles.append(h)
            for li in TEXT_LAYERS:
                stores[f"layer{li}"], h = hook_outputs(te.layers[li], first)
                handles.append(h)
            n = seg.shape[0]
            try:
                hidden = te(
                    input_ids=seg[None],
                    attention_mask=torch.ones(1, n, dtype=torch.long),
                    position_ids=torch.arange(n)[None],
                ).last_hidden_state
            finally:
                for h in handles:
                    h.remove()
            proj = model.text_encoder_proj(hidden)
            prefix = f"text_{branch}_seg{si}"
            for key, store in stores.items():
                assert len(store) == 1
                out.put(f"{prefix}_{key}", store[0][0])
            out.put(f"{prefix}_final_norm", hidden[0])
            out.put(f"{prefix}_proj", proj[0])
            proj_rows.append(proj[0].float())
        standalone = torch.cat(proj_rows)
        notes[branch] = float((standalone - text_rows).abs().max())
    return {"max_abs_diff_standalone_vs_pipeline_text_rows": notes}


@torch.inference_mode()
def backbone_prefill_goldens(model, specs: dict, out: TensorDir) -> dict[str, torch.Tensor]:
    bb = model.backbone_model
    logits_by_branch = {}
    for branch, spec in specs.items():
        embeds = model._merge_input_ids_with_input_values(**spec)["inputs_embeds"]
        s = embeds.shape[1]
        stores, handles = {}, []
        for li in BACKBONE_LAYERS:
            stores[f"layer{li}"], h = hook_outputs(bb.layers[li], first)
            handles.append(h)
        try:
            hidden = bb(
                inputs_embeds=embeds,
                attention_mask=torch.ones(1, s, dtype=torch.long),
                position_ids=torch.arange(s)[None],
                use_cache=True,
            ).last_hidden_state
        finally:
            for h in handles:
                h.remove()
        logits = model.lm_head(hidden[:, -1, :].float()).float()
        prefix = f"prefill_{branch}"
        out.put(f"{prefix}_inputs_embeds", embeds[0])
        for key, store in stores.items():
            out.put(f"{prefix}_{key}", store[0][0])
        out.put(f"{prefix}_final_norm", hidden[0])
        out.put(f"{prefix}_last_logits", logits[0])
        logits_by_branch[branch] = logits[0]
    return logits_by_branch


def store_stream(out: TensorDir, prefix: str, result: dict, sampled: bool) -> dict:
    out.i32(f"{prefix}_codes", result["codes"])
    for key in ("backbone_hidden", "backbone_logits", "backbone_logits_cfg", "depth_logits", "depth_logits_cfg"):
        out.put(f"{prefix}_{key}", result[key])
    out.i32(f"{prefix}_backbone_token_own", result["backbone_token_own"])
    out.i32(f"{prefix}_depth_token_own", result["depth_token_own"])
    if sampled:
        out.put(f"{prefix}_u_backbone", result["u_backbone"])
        out.put(f"{prefix}_u_depth", result["u_depth"])
    codes = result["codes"].numpy()
    own_frames = np.concatenate(
        [result["backbone_token_own"].numpy()[: len(codes), None], result["depth_token_own"].numpy()], axis=1
    )
    diverged = np.nonzero((own_frames != codes).any(axis=1))[0]
    return {
        "n_frames": result["n_frames"],
        "backbone_steps": int(result["backbone_logits"].shape[0]),
        "branches": result["branches"],
        "stopped_on_eos": result["stopped_on_eos"],
        "wall_s": round(result["wall_s"], 1),
        "first_frame_where_own_choice_differs": int(diverged[0]) if len(diverged) else None,
    }


# ---------------------------------------------------------------- codec


@torch.inference_mode()
def codec_full_forward(decoder, codes: torch.Tensor) -> dict[str, torch.Tensor]:
    """Q:869-884 step by step. codes: [N,16]."""
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
    reference = decoder(x)[0, 0]
    assert torch.equal(reference, stages["pcm_full"]), "replicated decoder forward drifted"
    return stages


@torch.inference_mode()
def codec_stream(atok, codes: torch.Tensor, chunk_frames: int) -> torch.Tensor:
    rt = MultiRequestStreamRuntime(
        atok,
        QwenStreamRuntimeConfig(
            chunk_frames=chunk_frames,
            num_lanes=1,
            max_active_reqs=1,
            fast=False,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
    )
    rt.open_request("golden", reset=True, is_first_decode=True)
    pieces = []
    # Same call pattern as the generation loop: one call per chunk.
    for start in range(0, codes.shape[0], chunk_frames):
        chunk = codes[start : start + chunk_frames].T[None].long()
        pieces.append(rt.decode_request_chunk("golden", chunk, reset=start == 0).reshape(-1))
    rt.close_request("golden")
    return torch.cat(pieces)


def per_frame_max_diff(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    n = min(a.numel(), b.numel()) // SAMPLES_PER_FRAME
    d = (a[: n * SAMPLES_PER_FRAME] - b[: n * SAMPLES_PER_FRAME]).abs().reshape(n, -1)
    return d.max(dim=1).values.numpy()


def stream_check(atok, codes: torch.Tensor) -> dict:
    full = codec_full_forward(atok.model.decoder, codes)["pcm_full"]
    s1 = codec_stream(atok, codes, 1)
    s2 = codec_stream(atok, codes, 2)
    d1 = per_frame_max_diff(full, s1)
    d2 = per_frame_max_diff(full, s2)

    def summary(d):
        big = np.nonzero(d > 1e-4)[0]
        return {
            "max_abs_frames_lt72": float(d[:72].max()),
            "max_abs_frames_ge72": float(d[72:].max()) if len(d) > 72 else None,
            "first_frame_over_1e-4": int(big[0]) if len(big) else None,
            "per_frame_max_abs": [float(f"{x:.3g}") for x in d],
        }

    fixed = KV.StaticShiftKVCache.update
    KV.StaticShiftKVCache.update = ORIGINAL_KV_UPDATE
    try:
        racy_a = codec_stream(atok, codes, 1)
        racy_b = codec_stream(atok, codes, 1)
    finally:
        KV.StaticShiftKVCache.update = fixed
    return {
        "n_frames": int(codes.shape[0]),
        "torch_threads": torch.get_num_threads(),
        "stream1_vs_full": summary(d1),
        "stream2_vs_full": summary(d2),
        "unpatched_kv_shift": {
            "stream1_rerun_max_abs": float((racy_a - racy_b).abs().max()),
            "stream1_vs_full": summary(per_frame_max_diff(full, racy_a)),
        },
    }


def codec_goldens(atok, codes: torch.Tensor, out: TensorDir) -> dict:
    stages = codec_full_forward(atok.model.decoder, codes)
    out.i32("codes", codes)
    for key, value in stages.items():
        out.put(key, value)
    s1 = codec_stream(atok, codes, 1)
    s2 = codec_stream(atok, codes, 2)
    out.put("pcm_stream1", s1)
    out.put("pcm_stream2", s2)
    full = stages["pcm_full"]
    return {
        "max_abs_stream1_vs_full": float((s1 - full).abs().max()),
        "max_abs_stream2_vs_full": float((s2 - full).abs().max()),
        "pcm_full": full,
        "pcm_stream2": s2,
    }


@torch.inference_mode()
def encoder_goldens(atok, out: TensorDir) -> torch.Tensor:
    wav, sr = sf.read(REF_CLIP, always_2d=True, dtype="float32")
    wav = np.mean(wav, axis=1)
    assert sr == SAMPLE_RATE
    enc = atok.model.encoder
    stores, handles = {}, []
    for name, module, pick in (
        ("seanet", enc.encoder, lambda o: o),
        ("transformer", enc.encoder_transformer, first),
        ("downsample", enc.downsample, lambda o: o),
    ):
        stores[name], h = hook_outputs(module, pick)
        handles.append(h)
    try:
        codes = torch.as_tensor(atok.encode(wav, sr=sr)["audio_codes"][0]).long()
    finally:
        for h in handles:
            h.remove()
    out.put("input_samples", wav)
    for name, store in stores.items():
        assert len(store) == 1, (name, len(store))
        out.put(name, store[0][0])
    out.i32("codes", codes)
    return codes


# ---------------------------------------------------------------- dtype audit


@contextmanager
def dtype_audit():
    seen: dict[str, set[str]] = {}

    def fn(module, _inp, out):
        for t in out if isinstance(out, (tuple, list)) else [out]:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                seen.setdefault(str(t.dtype).replace("torch.", ""), set()).add(type(module).__name__)

    handle = torch.nn.modules.module.register_module_forward_hook(fn)
    try:
        yield seen
    finally:
        handle.remove()


# ---------------------------------------------------------------- main


def load(precision: str):
    tok, model, atok = load_runtime(WEIGHTS, device="cpu", attn_implementation="eager")
    update_generation_config_for_breeze(model)
    scale = model.text_encoder.embed_tokens.embed_scale
    info = {"embed_scale_buffer": [float(scale), str(scale.dtype).replace("torch.", "")]}
    if precision == "fp32":
        model.float()
    # The buffer is created as bf16 (34.0), so widening keeps 34.0; record it.
    info["embed_scale_used"] = float(model.text_encoder.embed_tokens.embed_scale)
    # BG:70 casts lm_head to fp32 when the runtime is built; do it before the
    # module goldens so they see the same head as the stream.
    model.lm_head.float()
    model.depth_decoder.codebooks_head.weight.data = model.depth_decoder.codebooks_head.weight.data.float()
    return tok, model, atok, info


def make_ref_clip(audio: np.ndarray, source: str) -> None:
    clip = audio[: REF_CLIP_FRAMES * SAMPLES_PER_FRAME]
    write_wav(REF_CLIP, clip)
    REF_CLIP.with_suffix(".json").write_text(
        json.dumps(
            {
                "source": source,
                "samples": int(len(clip)),
                "seconds": len(clip) / SAMPLE_RATE,
                "transcript": DESIGN_EN_TEXT,
                "transcript_note": "the design_en text; check that the clip holds all of it (see README)",
            },
            indent=1,
        )
        + "\n"
    )


def run_case(name: str, precision: str, tok, model, atok, load_info: dict) -> None:
    case = CASES[name]
    case_dir = GOLDENS / name
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.manual_seed(SEED)
    inputs = prepare_inputs(
        tok,
        atok,
        model,
        [build_request(case)],
        get_template(case["template"]),
        guidance_scale=case["cfg_scale"],
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    specs = branch_specs(inputs)
    out = TensorDir(case_dir, precision)
    run: dict = {"precision": precision, **load_info}

    with dtype_audit() as seen:
        run["text_encoder"] = text_encoder_goldens(model, specs, out)
        unpadded_logits = backbone_prefill_goldens(model, specs, out)

        streams = {}
        for mode in ("greedy", "sampled"):
            forced = None
            if precision == "bf16":
                fp32 = json.loads((case_dir / "manifest.json").read_text())["runs"]["fp32"]
                entry = fp32["tensors"][f"{mode}_codes"]
                codes = np.fromfile(case_dir / entry["file"], dtype="<i4").reshape(entry["shape"])
                forced = (codes, fp32[mode]["stopped_on_eos"])
            result = StreamRecorder(mode, forced).run(model, atok, tok, inputs)
            run[mode] = store_stream(out, mode, result, mode == "sampled")
            if forced is not None:
                assert np.array_equal(result["codes"].numpy(), forced[0]), "forced replay lost the stream"
            streams[mode] = result
            write_wav(case_dir / f"{precision}_{mode}.wav", result["audio"])
            print(f"[{name} {precision}] {mode}: {run[mode]}", flush=True)
    run["dtype_audit"] = {k: sorted(v) for k, v in seen.items()}

    batched = streams["greedy"]["backbone_logits"][0]
    run["prefill_last_logits_max_abs_unpadded_vs_batched"] = {
        branch: float((logits - batched[i]).abs().max()) for i, (branch, logits) in enumerate(unpadded_logits.items())
    }
    run["tensors"] = out.index
    run["wall_s"] = round(time.time() - started, 1)

    sections = {"runs": {precision: run}}
    if precision == "fp32":
        sections.update(
            case=name,
            template=case["template"],
            text=case["text"],
            instruction=case.get("instruction"),
            cfg_scale=case["cfg_scale"],
            seed=SEED,
            frame_cap=FRAME_CAP,
            sampling={
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 1.0,
                "repetition_penalty": 1.1,
                "suppressed": [2048, 2049, 2050],
            },
            tokens=token_manifest(tok, specs),
        )
        if case.get("ref"):
            sections["ref"] = json.loads(REF_CLIP.with_suffix(".json").read_text()) | {
                "wav": os.path.relpath(REF_CLIP, case_dir)
            }
            enc_out = TensorDir(case_dir, "encoder")
            ref_codes = encoder_goldens(atok, enc_out)
            pipeline_codes = inputs["input_values"][0].long()
            assert torch.equal(ref_codes, pipeline_codes), "encoder golden codes differ from prepare_inputs"
            sections["encoder"] = {"tensors": enc_out.index}
        codec_out = TensorDir(case_dir, "codec")
        codec = codec_goldens(atok, streams["greedy"]["codes"], codec_out)
        own = torch.from_numpy(streams["greedy"]["audio"])
        sections["codec"] = {
            "source": "fp32 greedy codes",
            "max_abs_stream1_vs_full": codec["max_abs_stream1_vs_full"],
            "max_abs_stream2_vs_full": codec["max_abs_stream2_vs_full"],
            "max_abs_stream2_vs_generation_audio": float((codec["pcm_stream2"] - own).abs().max()),
            "tensors": codec_out.index,
        }
        write_wav(case_dir / "codec_full_forward.wav", codec["pcm_full"].numpy())
    update_manifest(case_dir, **sections)
    print(f"[{name} {precision}] done in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--stream-check", action="store_true", help="tile design_en greedy codes past 72 frames and compare codec streaming modes")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    fix_codec_kv_shift()
    GOLDENS.mkdir(parents=True, exist_ok=True)
    tok, model, atok, load_info = load(args.precision)
    print("load:", load_info, flush=True)
    for name in args.cases.split(","):
        run_case(name, args.precision, tok, model, atok, load_info)
        if name == "design_en" and args.precision == "fp32":
            fp32 = json.loads((GOLDENS / name / "manifest.json").read_text())["runs"]["fp32"]
            pick = "sampled" if fp32["sampled"]["stopped_on_eos"] else "greedy"
            audio, _ = sf.read(GOLDENS / name / f"fp32_{pick}.wav", dtype="float32")
            make_ref_clip(audio, f"design_en/fp32_{pick}.wav (2-frame streaming output)")
    if args.stream_check:
        manifest = json.loads((GOLDENS / "design_en" / "manifest.json").read_text())
        entry = manifest["codec"]["tensors"]["codes"]
        codes = torch.from_numpy(
            np.fromfile(GOLDENS / "design_en" / entry["file"], dtype="<i4").reshape(entry["shape"])
        ).long()
        tiled = codes.repeat((150 + len(codes) - 1) // len(codes), 1)[:150]
        result = stream_check(atok, tiled)
        (GOLDENS / "codec_stream_check.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: v for k, v in result.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
