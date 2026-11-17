#!/usr/bin/env python3
# Copyright 2026 Peter Marelas
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced depth-decoder yardstick for weight quantization.

    .venv/bin/python tools/quant/depth_quant.py band
    .venv/bin/python tools/quant/depth_quant.py sweep [--only NAME ...]
    .venv/bin/python tools/quant/depth_quant.py checksum

Runs its own fp32 depth decoder (12 layers, RMSNorm, grouped-query attention with the llama3
RoPE table, SwiGLU) on the golden backbone hidden states and codes of the five reference cases
(.audit/goldens, or BREEZE_GOLDENS), and scores the logits a quantized copy of the weights gives
against the fp32 goldens. `band` first proves the decoder reproduces the fp32 goldens, then
prints the drift of the reference's own bf16 run, which is the unit every quantization error is
read in. Nothing is written to disk: quantized weights exist only in this process.

Scores are taken where the sampler looks: the guidance-combined logits with ids 2048 to 2050
suppressed, temperature 0.9, top-k 50.
  flip   probability that one uniform draws a different code from the two distributions
  tv     total variation distance
  kl     KL(reference || arm), the arm's probabilities floored at 1e-12
  top1   share of draws whose most likely code agrees
"""
import argparse, json, os, sys, zlib
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = Path(os.environ.get("BREEZE_GOLDENS", ROOT / ".audit/goldens"))
CASES = ["design_en", "plain_en", "design_zh", "clone_en", "direction_en"]
LAYERS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, VOCAB, BOOKS = 12, 1024, 8, 2, 128, 2051, 16
PREFIX = "depth_decoder.model"
MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]


def weights_dir():
    if "BREEZE_WEIGHTS" in os.environ:
        return Path(os.environ["BREEZE_WEIGHTS"])
    snaps = sorted((Path.home() / ".cache/huggingface/hub/models--BreezeBlue--Breeze-TTS-2/snapshots").glob("*"))
    return snaps[-1]


def load_depth():
    """Every depth-decoder tensor as fp32 (the bf16 values widened exactly)."""
    out = {}
    for shard in sorted(weights_dir().glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                if name.startswith("depth_decoder."):
                    out[name] = f.get_tensor(name).float()
    return out


def rope_tables(dtype):
    inv = 1.0 / (5e5 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    low_wavelen, high_wavelen = 16 / 0.001953125, 16 / 0.0078125
    wavelen = 2 * torch.pi / inv
    scaled = torch.where(wavelen > low_wavelen, inv / 32, inv)
    smooth = (16 / wavelen - 0.001953125) / (0.0078125 - 0.001953125)
    mid = (1 - smooth) * scaled / 32 + smooth * scaled
    inv = torch.where((wavelen >= high_wavelen) & (wavelen <= low_wavelen), mid, scaled)
    angle = torch.arange(BOOKS, dtype=torch.float32)[:, None] * inv[None]
    angle = torch.cat([angle, angle], dim=-1)
    return angle.cos().to(dtype), angle.sin().to(dtype)


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


def rmsnorm(x, w, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@torch.no_grad()
def forward(W, hidden, codes, dtype=torch.float32):
    """hidden [N, B, 2048], codes [N, 16] -> logits [N, 15, B, 2051] (fp32 heads)."""
    n, b, _ = hidden.shape
    c = lambda name: W[name].to(dtype)
    table = W[f"{PREFIX}.embed_tokens.weight"]
    ids = codes[:, :15].long() + torch.arange(15)[None] * VOCAB
    emb = table[ids][:, None].expand(n, b, 15, 2048)
    x = torch.cat([hidden[:, :, None], emb], dim=2).reshape(n * b, BOOKS, 2048).to(dtype)
    x = x @ c(f"{PREFIX}.inputs_embeds_projector.weight").T
    cos, sin = rope_tables(dtype)
    mask = torch.full((BOOKS, BOOKS), float("-inf")).triu(1).to(dtype)
    s = x.shape[0]
    for i in range(LAYERS):
        p = f"{PREFIX}.layers.{i}"
        h = rmsnorm(x, c(f"{p}.input_layernorm.weight"))
        q = (h @ c(f"{p}.self_attn.q_proj.weight").T).view(s, BOOKS, HEADS, HEAD_DIM).transpose(1, 2)
        k = (h @ c(f"{p}.self_attn.k_proj.weight").T).view(s, BOOKS, KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = (h @ c(f"{p}.self_attn.v_proj.weight").T).view(s, BOOKS, KV_HEADS, HEAD_DIM).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        k = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
        v = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
        att = torch.softmax((q @ k.transpose(-1, -2)) / HEAD_DIM ** 0.5 + mask, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(s, BOOKS, HEADS * HEAD_DIM)
        x = x + o @ c(f"{p}.self_attn.o_proj.weight").T
        h = rmsnorm(x, c(f"{p}.post_attention_layernorm.weight"))
        g = torch.nn.functional.silu(h @ c(f"{p}.mlp.gate_proj.weight").T) * (h @ c(f"{p}.mlp.up_proj.weight").T)
        x = x + g @ c(f"{p}.mlp.down_proj.weight").T
    x = rmsnorm(x, c(f"{PREFIX}.norm.weight")).float()
    heads = W["depth_decoder.codebooks_head.weight"]
    logits = torch.einsum("spk,pkv->spv", x[:, 1:], heads)
    return logits.view(n, b, 15, VOCAB).transpose(1, 2).contiguous()


def load_streams():
    """Every golden stream: name, cfg scale, fp32 inputs, fp32 and bf16 depth logits."""
    streams = []
    for case in CASES:
        d = GOLDENS / case
        m = json.loads((d / "manifest.json").read_text())
        for mode in ("greedy", "sampled"):
            def get(precision, key):
                e = m["runs"][precision]["tensors"][f"{mode}_{key}"]
                dt = "<i4" if e["dtype"] == "int32" else "<f4"
                return torch.from_numpy(np.fromfile(d / e["file"], dtype=dt).reshape(e["shape"]).copy())
            streams.append(dict(name=f"{case}/{mode}", cfg=float(m["cfg_scale"]), hidden=get("fp32", "backbone_hidden"),
                                codes=get("fp32", "codes"), fp32=get("fp32", "depth_logits"), bf16=get("bf16", "depth_logits")))
    return streams


def sampler_view(logits, cfg):
    """[N, 15, B, V] raw logits -> the distribution each draw samples from, [N * 15, V] float64."""
    x = logits.double()
    x = x[:, :, 0] if x.shape[2] == 1 else x[:, :, 1] + cfg * (x[:, :, 0] - x[:, :, 1])
    x = x.reshape(-1, VOCAB).clone()
    x[:, 2048:] = float("-inf")
    x = x / 0.9
    kth = x.topk(50, dim=-1).values[:, -1:]
    x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
    return torch.softmax(x, dim=-1)


def score(ref, arm, cfg):
    """Per-draw scores of `arm` against `ref`, both [N, 15, B, V] raw logits."""
    p, q = sampler_view(ref, cfg), sampler_view(arm, cfg)
    cp, cq = p.cumsum(-1), q.cumsum(-1)
    lo = torch.maximum(cp - p, cq - q)
    hi = torch.minimum(cp, cq)
    flip = 1 - (hi - lo).clamp(min=0).sum(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    kl = torch.where(p > 0, p * (p.clamp(min=1e-300).log() - q.clamp(min=1e-12).log()), torch.zeros_like(p)).sum(-1)
    top1 = (p.argmax(-1) == q.argmax(-1)).double()
    a, b = ref.reshape(-1, VOCAB).double(), arm.reshape(-1, VOCAB).double()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return dict(flip=flip, tv=tv, kl=kl, top1=top1, cos=cos, maxabs=(a - b).abs().max(-1).values)


def summarize(parts):
    """One row over every draw of every stream."""
    cat = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    q = lambda t, f: float(torch.quantile(t, f))
    return dict(flip=float(cat["flip"].mean()), flip_p99=q(cat["flip"], 0.99), tv=float(cat["tv"].mean()), tv_p99=q(cat["tv"], 0.99),
                kl=float(cat["kl"].mean()), kl_p99=q(cat["kl"], 0.99), top1=float(cat["top1"].mean()),
                cos_min=float(cat["cos"].min()), maxabs=float(cat["maxabs"].max()), draws=len(cat["flip"]))


COLUMNS = ["flip", "flip_p99", "tv", "tv_p99", "kl", "kl_p99", "top1", "cos_min", "maxabs"]


def row(name, s, extra=""):
    vals = "\t".join(f"{s[c]:.6f}" if c in ("top1", "cos_min") else f"{s[c]:.3e}" for c in COLUMNS)
    return f"{name}\t{vals}\t{extra}"


def header():
    return "arm\t" + "\t".join(COLUMNS) + "\tnote"


def round_half_away(x):
    """Ties away from zero, the rule Swift's .rounded() uses; torch.round is half to even."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


LANES, RUN = 32, 16


def to_stripes(w):
    """[N, K] -> [N, K] with each lane's weights contiguous: lane l of a 32-lane simdgroup that
    loads 16 int8 at a time reads k = 512 i + 16 l ... + 15 for i = 0, 1, ..."""
    n, k = w.shape
    return w.view(n, k // (LANES * RUN), LANES, RUN).transpose(1, 2).reshape(n, k)


def from_stripes(w):
    n, k = w.shape
    return w.view(n, LANES, k // (LANES * RUN), RUN).transpose(1, 2).reshape(n, k)


def pow2_cover(amax):
    """The smallest power of two s with amax <= 127 s, from the float's own exponent (no log)."""
    m, x = torch.frexp(amax)
    e = torch.where(m <= 127 / 128, x - 7, x - 6)
    return torch.where(amax > 0, torch.exp2(e.float()), torch.ones_like(amax))


def quantize(w, block, scale="pow2", clip="max", stripes=False):
    """Symmetric int8 over blocks of `block` weights along K of w [N, K] (0: the whole row).

    Returns (q int8 [N, K], scales fp32 [N, K / block]). scale "pow2" restricts scales to powers
    of two, so q * scale is exact in bf16; "free" keeps max / 127 rounded to fp16. clip "max"
    covers the largest weight; "mse" also tries smaller scales that clip it and keeps the one
    with the least squared error. With `stripes` a block is `block` weights of one lane's
    stripe (see to_stripes) and block 0 is the whole stripe, K / 32 weights; the scales are
    then in stripe order, [N, 32 * blocks per stripe].
    """
    n, k = w.shape
    if stripes:
        q, s = quantize(to_stripes(w), block or k // LANES, scale, clip)
        return from_stripes(q), s
    block = block or k
    v = w.view(n, k // block, block)
    amax = v.abs().amax(-1, keepdim=True)

    def candidates():
        if scale == "pow2":
            cover = pow2_cover(amax)
            for shift in ([1.0] if clip == "max" else [1.0, 0.5, 0.25]):
                yield cover * shift
        else:
            for f in ([1.0] if clip == "max" else [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]):
                yield (amax * f / 127).half().float()

    best_q = best_s = best_err = None
    for s in candidates():
        q = round_half_away(torch.where(s > 0, v / s, torch.zeros_like(v))).clamp(-127, 127)
        err = ((q * s - v) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_q, best_s, best_err = q, s.expand_as(amax).clone(), err
        else:
            take = err < best_err
            best_q = torch.where(take, q, best_q)
            best_s = torch.where(take, s, best_s)
            best_err = torch.where(take, err, best_err)
    return best_q.view(n, k).to(torch.int8), best_s.view(n, k // block)


def dequantize(q, s, stripes=False):
    n, k = q.shape
    if stripes:
        return from_stripes(dequantize(to_stripes(q), s))
    return (q.view(n, s.shape[1], -1).float() * s[:, :, None]).view(n, k)


def quantized_copy(W, tensors, **scheme):
    """W with the named per-layer tensors replaced by their dequantized int8 form, the weight
    bytes of the scheme (scales as fp16, or one exponent byte when pow2), and the relative
    squared error over the replaced tensors."""
    out = dict(W)
    nbytes, err, ref = 0, 0.0, 0.0
    for i in range(LAYERS):
        for t in tensors:
            name = f"{PREFIX}.layers.{i}.{t}.weight"
            q, s = quantize(W[name], **scheme)
            out[name] = dequantize(q, s, scheme.get("stripes", False))
            nbytes += q.numel() + s.numel() * (1 if scheme.get("scale", "pow2") == "pow2" else 2)
            err += float(((out[name] - W[name]) ** 2).sum())
            ref += float((W[name] ** 2).sum())
    return out, nbytes, err / ref


def bf16_bytes(W, tensors):
    return sum(W[f"{PREFIX}.layers.{i}.{t}.weight"].numel() * 2 for i in range(LAYERS) for t in tensors)


def evaluate(W, streams, dtype=torch.float32):
    return summarize([score(s["fp32"], forward(W, s["hidden"], s["codes"], dtype), s["cfg"]) for s in streams])


def cmd_band(_):
    W, streams = load_depth(), load_streams()
    print(header())
    own = evaluate(W, streams)
    print(row("own fp32 decoder", own, f"{own['draws']} draws; must match the fp32 goldens"))
    if own["cos_min"] < 0.99999 or own["flip"] > 1e-4:
        sys.exit("the decoder here does not reproduce the fp32 goldens")
    print(row("reference bf16 (golden)", summarize([score(s["fp32"], s["bf16"], s["cfg"]) for s in streams]),
              "bf16 backbone and depth decoder, fp32 heads, teacher-forced on the fp32 codes"))
    print(row("depth-only bf16", evaluate(W, streams, torch.bfloat16), "fp32 hidden states in, bf16 arithmetic in the depth decoder only"))
    for s in streams:
        print(row("  golden bf16 " + s["name"], summarize([score(s["fp32"], s["bf16"], s["cfg"])])))


SCHEMES = {}
for _tensors, _tname in ((MLP, "mlp"), (MLP + ATTN, "mlp+attn"), (ATTN, "attn")):
    for _block in (0, 128, 64, 32, 16):
        for _scale in ("pow2", "free"):
            for _clip in ("max", "mse"):
                SCHEMES[f"{_tname}/b{_block or 'row'}/{_scale}/{_clip}"] = (_tensors, dict(block=_block, scale=_scale, clip=_clip))


for _tensors, _tname in ((MLP, "mlp"), (MLP + ATTN, "mlp+attn"), (ATTN, "attn")):
    for _block in (0, 64, 32):
        for _scale in ("pow2", "free"):
            SCHEMES[f"{_tname}/stripe{_block or 'all'}/{_scale}/max"] = (_tensors, dict(block=_block, scale=_scale, clip="max", stripes=True))


def cmd_sweep(args):
    W, streams = load_depth(), load_streams()
    total = bf16_bytes(W, MLP + ATTN)
    print(header())
    for name, (tensors, scheme) in SCHEMES.items():
        if args.only and not any(o in name for o in args.only):
            continue
        Wq, nbytes, rel = quantized_copy(W, tensors, **scheme)
        after = total - bf16_bytes(W, tensors) + nbytes
        print(row(name, evaluate(Wq, streams), f"layer bytes {after / 1e6:.1f} MB of {total / 1e6:.1f} ({after / total:.3f}); weight SNR {-10 * np.log10(rel):.1f} dB"), flush=True)


def cmd_checksum(args):
    """CRC32 of q and of the scales' bit patterns for one tensor, to compare with the Swift quantizer."""
    W = load_depth()
    for name in (f"{PREFIX}.layers.0.mlp.gate_proj.weight", f"{PREFIX}.layers.11.mlp.down_proj.weight", f"{PREFIX}.layers.5.self_attn.o_proj.weight"):
        q, s = quantize(W[name], block=args.block, scale=args.scale, clip=args.clip, stripes=args.stripes)
        print(f"{name}\tq crc32 {zlib.crc32(q.numpy().tobytes()):08x}\tscales crc32 {zlib.crc32(s.numpy().astype('<f4').tobytes()):08x}")


def main():
    torch.manual_seed(0)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("band").set_defaults(fn=cmd_band)
    sw = sub.add_parser("sweep")
    sw.add_argument("--only", nargs="*")
    sw.set_defaults(fn=cmd_sweep)
    ck = sub.add_parser("checksum")
    ck.add_argument("--block", type=int, default=32)
    ck.add_argument("--scale", default="pow2")
    ck.add_argument("--clip", default="max")
    ck.add_argument("--stripes", action="store_true")
    ck.set_defaults(fn=cmd_checksum)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
