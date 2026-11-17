<!-- Copyright 2026 Peter Marelas -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Reference goldens

These scripts run the upstream Breeze TTS 2 Python reference on CPU and write the tensors that the Swift tests compare against. Nothing here ships. The goldens live in `.audit/goldens/`, which is gitignored.

## Run

Use the repo venv (`.venv`: torch 2.9.1, transformers 4.57.3, qwen-tts 0.1.1). Everything runs on CPU. Do not use MPS.

```sh
export BREEZE_WEIGHTS=~/.cache/huggingface/hub/models--BreezeBlue--Breeze-TTS-2/snapshots/<rev>
export BREEZE_REF_SRC=/path/to/breeze-tts-reference-clone
.venv/bin/python tools/reference/make_goldens.py --precision fp32 --stream-check   # about 12 min
.venv/bin/python tools/reference/make_goldens.py --precision bf16                  # about 7 min, replays the fp32 codes
.venv/bin/python tools/reference/check_goldens.py
.venv/bin/python tools/reference/verify_checkpoint.py                              # spec claims vs checkpoint and loaded modules
```

`BREEZE_GOLDENS` overrides the output directory. The fp32 pass must run first. It writes the reference clip that the clone cases read, and the bf16 pass replays its codes.

## What runs

The harness drives the canonical path, `FastBreezeStreamingRuntime.iter_audio_chunks` with every `fast_*` flag off, through three small changes:

- `CpuStreamingRuntime` swallows the single `fast streaming requires a CUDA device` error at the end of `__init__`. Every attribute is already set at that point.
- `StaticShiftKVCache.update` in the codec stream runtime is replaced by a version that copies the shifted window out before writing. The upstream overlapping in-place `copy_` races on a multi-threaded CPU. Two decodes of the same codes then differ by up to 4.6e-3 from frame 24 on. `codec_stream_check.json` keeps the unpatched numbers.
- The two sampling sites are wrapped for recording, as described below. Their arithmetic is unchanged.

## Precisions

- `fp32` is the canonical run. It loads the bf16 checkpoint and calls `model.float()`, so the weights are the bf16 values widened exactly. Every module output in this run is float32; `runs.fp32.dtype_audit` lists the dtypes seen by a global forward hook. The text-encoder `embed_scale` buffer is created in bf16 (34.0) and stays 34.0 after widening. This is the tight reference for a port that computes in fp32 with bf16 weights.
- `bf16` is the model as shipped. `lm_head` and `codebooks_head` run in fp32, as the reference casts them (BG:70, DG:161). Everything else runs in bf16 on CPU kernels. This run is teacher-forced onto the fp32 codes: each sampler runs normally, its own choice is recorded in `*_token_own`, and then the fp32 token replaces it. Per-frame bf16 tensors are therefore comparable with fp32 for the whole stream. `runs.bf16.<mode>.first_frame_where_own_choice_differs` says where a free bf16 run would leave the fp32 stream.
- The codec always runs in fp32, so the `codec/` and `encoder/` goldens exist once per case.

## Cases

| Case | Template | cfg | Notes |
|---|---|---|---|
| `design_en` | `tts_instruction` | 4 | text and instruction from the task brief |
| `plain_en` | `tts_plain` | 1 | |
| `design_zh` | `tts_instruction` | 4 | text starts with the `[笑]` event |
| `clone_en` | `ref_clone_tata` | 1 | reference clip below |
| `direction_en` | `ref_edit_tata` | 4 | reference clip below |

The seed is 42 for every case (`torch.manual_seed`, and the uniforms). Sampling uses temperature 0.9, top-k 50, top-p 1.0, and repetition penalty 1.1 on the backbone. Ids 2048, 2049 and 2050 are suppressed. Generation is capped at 120 frames. Every case stopped on EOS well before the cap.

The reference clip is `ref_clip_en.wav`, described in `ref_clip_en.json`. It is the 2-frame streaming output of the `design_en` fp32 sampled run, written as 24 kHz mono PCM16, which is how `infer.py` writes audio. That utterance is 34 frames (2.72 s), shorter than the planned 4 s, so the clip is the whole utterance. Its transcript is the `design_en` text. Whisper small.en reads it back as exactly that text.

## Layout

```
.audit/goldens/
  ref_clip_en.wav, ref_clip_en.json
  codec_stream_check.json
  <case>/
    manifest.json
    fp32/  bf16/   per-precision tensors
    codec/         decoder goldens from the fp32 greedy codes
    encoder/       clone and direction only
    fp32_greedy.wav, fp32_sampled.wav, bf16_greedy.wav, bf16_sampled.wav, codec_full_forward.wav
```

Every tensor file is raw little-endian, row-major, with no header. `.f32` files are float32 and `.i32` files are int32. The bf16 run's activations are stored widened to float32, which is exact.

`manifest.json` holds:

- `case`, `template`, `text`, `instruction`, `cfg_scale`, `seed`, `frame_cap`, `sampling`.
- `tokens.cond` and `tokens.uncond` (uncond only when cfg != 1). Each has `input_ids` (the full backbone ids), `text_ids_mask` (0/1 per position), `text_ids_len`, `segments` (the ids of each text segment, which is what the text encoder sees), and `rendered` (the decoded prompt string).
- `ref` for clone cases. It holds the clip transcript, the sample count and the path.
- `runs.fp32` and `runs.bf16`. Each holds `tensors` (`{name: {file, shape, dtype}}`), the per-stream summaries `greedy` and `sampled` (`n_frames`, `backbone_steps`, `branches`, `stopped_on_eos`, `wall_s`, `first_frame_where_own_choice_differs`), `dtype_audit`, `text_encoder.max_abs_diff_standalone_vs_pipeline_text_rows`, `prefill_last_logits_max_abs_unpadded_vs_batched`, `embed_scale_*`, and `wall_s`.
- `codec.tensors` and `encoder.tensors`, plus codec agreement numbers.

## Tensors per precision (`<case>/<precision>/`)

`B` is the number of branches: 2 with CFG (row 0 cond, row 1 uncond), otherwise 1. `L` is the segment length in tokens, `S` the branch prompt length, `N` the number of frames, and `M` the number of backbone sampling steps. `M = N + 1` when the run stopped on EOS, and the last step is the one that produced EOS.

**Text encoder.** One set per branch and text segment, with names `text_<branch>_seg<i>_<stage>`. Each segment runs alone at batch 1 with positions `0..L-1`. In fp32 the projected rows equal the reference's own bucketed, right-padded batch exactly. In bf16 they differ by up to 0.125 when two segments share a padded bucket (clone and direction), because the padded matmul shapes round differently.

| stage | shape | meaning |
|---|---|---|
| `embed` | [L, 1152] | embedding times 34.0 |
| `layer0`, `layer4`, `layer5`, `layer25` | [L, 1152] | residual stream after that layer (layer 5 is the first full-attention layer) |
| `final_norm` | [L, 1152] | after `text_encoder.norm` |
| `proj` | [L, 2048] | `text_encoder_proj` output |

**Backbone prefill.** Names are `prefill_<branch>_<stage>`. Each branch runs alone, unpadded, at batch 1, with positions `0..S-1` and a plain causal mask.

| stage | shape | meaning |
|---|---|---|
| `inputs_embeds` | [S, 2048] | text rows, audio rows and the audio-eos row, as merged by the reference |
| `layer0`, `layer27` | [S, 2048] | residual after layer 0 and after the last layer, before the final norm |
| `final_norm` | [S, 2048] | `last_hidden_state` |
| `last_logits` | [2052] | `lm_head(final_norm[-1])` in fp32 |

The reference stream itself batches cond and uncond with left padding. Its prefill logits (`greedy_backbone_logits[0]`) differ from the unpadded ones by at most 2.7e-5 in fp32 and by up to 0.032 in bf16 (design_en uncond, 19 pad rows). See `prefill_last_logits_max_abs_unpadded_vs_batched`.

**Streams.** The prefix is `greedy_` or `sampled_`. In greedy mode both samplers run with `do_sample=False`, so the backbone takes the argmax after the repetition penalty and suppression, and the depth decoder takes the argmax of the CFG-combined logits after suppression.

| name | shape | dtype | meaning |
|---|---|---|---|
| `codes` | [N, 16] | i32 | frames as emitted: code 0 from the backbone, codes 1..15 from the depth decoder |
| `backbone_hidden` | [N, B, 2048] | f32 | backbone last hidden fed to the depth decoder for frame k. Row 0 is the prefill's last position. Row k is the output of feeding frame k-1 |
| `backbone_logits` | [M, B, 2052] | f32 | raw `lm_head` output per branch, in fp32 |
| `backbone_logits_cfg` | [M, 2052] | f32 | sampler input: `uncond + s*(cond-uncond)` in fp32, before the penalty and suppression. Equals `backbone_logits[:,0]` when B = 1 |
| `backbone_token_own` | [M] | i32 | the sampler's own pick. The last entry is 2051 when the run stopped on EOS |
| `depth_logits` | [N, 15, B, 2051] | f32 | raw `codebooks_head` output. `[k, c]` is the logits for codebook c+1 of frame k |
| `depth_logits_cfg` | [N, 15, 2051] | f32 | CFG-combined, before suppression |
| `depth_token_own` | [N, 15] | i32 | the depth sampler's own pick |
| `u_backbone` | [M] | f32 | sampled only: the uniform for backbone draw m |
| `u_depth` | [N, 15] | f32 | sampled only: the uniform for the draw of codebook c+1 of frame k |

In fp32, `*_token_own` equals `codes`. In bf16 it is what bf16 would have chosen.

**Injected sampling.** In the sampled streams `torch.multinomial` is replaced at both reference sites (`sampling.py:86` and `DepthDecoderGraph._cfg_sample`, DG:548). The reference's temperature, top-k, top-p, sort and scatter run unchanged, so the replacement sees the final probability vector `p` in natural token order. The rule:

```
cdf    = cumsum(float64(p))          # p is the float32 probability row
target = float64(u) * cdf[-1]
token  = first i with cdf[i] > target   # strict; zero-probability tokens are never chosen
         (if no i qualifies: the last i with p[i] > 0; no stored u reaches 1.0, so this never fired)
```

`check_goldens.py` replays both samplers in numpy from the stored fp32 tensors: CFG-combined logits, code-0 history, penalty, suppression, temperature, top-k 50 by threshold, softmax and the rule above. It reproduces every greedy and sampled token in all five cases.

The uniforms come from `numpy.random.Generator(PCG64(42))`: `random(121)` for the backbone, then `random((120, 15))` for the depth draws, each rounded to float32. The stored float32 values are the inputs, so a port does not need the generator. Draw order per request is backbone draw 0 after prefill, then for each frame 15 depth draws and one backbone draw. In greedy mode the depth graph still calls the real `torch.multinomial` and discards the result.

## Codec goldens (`<case>/codec/`)

These come from the fp32 greedy codes, `T = N` frames, batch dimension dropped. They come from one full forward of `Qwen3TTSTokenizerV2Decoder` (Q:869-884), replicated step by step and asserted bit-equal to `decoder(codes)`.

| name | shape |
|---|---|
| `codes` | [T, 16] i32 |
| `quantizer` | [512, T] |
| `pre_conv` | [1024, T] |
| `pre_transformer` | [T, 1024] (after its output_proj) |
| `upsample0`, `upsample1` | [1024, 2T], [1024, 4T] |
| `decoder0` | [1536, 4T] |
| `decoder1` .. `decoder4` | [768, 32T], [384, 160T], [192, 640T], [96, 1920T] |
| `decoder5` | [96, 1920T] (SnakeBeta) |
| `decoder6` | [1, 1920T] (before the clamp) |
| `pcm_full` | [1920T], clamped to [-1, 1] |
| `pcm_stream1` | [1920T], reference stream runtime with 1-frame chunks |
| `pcm_stream2` | [1920T], 2-frame chunks (the `infer.py` default); equals the generation run's own audio |

`codec_stream_check.json` decodes the design_en greedy codes tiled to 150 frames. With the KV fix, 1-frame streaming matches the full forward to 1.9e-6 on every frame. 2-frame streaming matches to 1.8e-6 before frame 72 and differs by up to 4.1e-3 from frame 72 on.

## Encoder goldens (`<case>/encoder/`, clone and direction)

| name | shape |
|---|---|
| `input_samples` | [n] f32, what `soundfile.read` returns for the PCM16 clip (24 kHz, so no resampling) |
| `seanet` | [512, T25] |
| `transformer` | [T25, 512] |
| `downsample` | [512, T12] |
| `codes` | [T12, 16] i32, equal to the codes `prepare_inputs` put into the prompt |
