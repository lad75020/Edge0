# Gemma 4 31B MLX

`gemma-4:31b-mlx` is Edge0's resident text-generation tier for
[`mlx-community/gemma-4-31b-4bit`](https://huggingface.co/mlx-community/gemma-4-31b-4bit),
an MLX 4-bit conversion of `google/gemma-4-31b` made with mlx-vlm 0.4.3.

## Current scope and licensing

Edge0 loads only the dense language model. Image, audio, and video inputs are
not wired through the Python, CLI, or server APIs. Preparation removes all 355
`vision_tower.*` tensors and all three `embed_vision.*` tensors while retaining
the language tensors and coherent safetensors metadata.

The adapted language architecture comes from MIT-licensed mlx-vlm 0.4.3; Edge0
does not add mlx-vlm as a dependency. Model weights are not distributed by
Edge0 and remain subject to the Gemma terms.

This is not a MoE model. It never installs streaming experts and has no
prerouter, LoRA, history staging, intra-layer staging, prefetch, or hot window.

## Architecture profile

| Setting | Value |
|---|---:|
| Layers | 60 |
| Hidden / MLP intermediate | 5376 / 21504 |
| Vocabulary | 262144, tied embeddings |
| Query heads | 32 |
| Sliding KV heads / head dimension | 16 / 256 |
| Global KV heads / head dimension | 4 / 512 |
| Pattern | five sliding layers, then one full layer |
| Sliding window | 1024 |
| Sliding RoPE | theta 10000 |
| Full RoPE | theta 1000000, proportional, 0.25 partial rotary |
| RMS epsilon | 1e-6 |
| Final logit softcap | 30 |
| Quantization | 4-bit affine, group size 64 |
| Server port | 8088 |

Full-attention layers use `attention_k_eq_v`; `hidden_size_per_layer_input`
and `num_kv_shared_layers` are both zero.

## Generation and prompt format

Defaults are temperature `1.0`, top-p `0.95`, top-k `64`, repetition penalty
`1.0`, EOS `(1,)`, and at most 2048 new tokens. The prepared directory receives
an attributed local text-only adaptation of the official Gemma 4 IT chat
template. It emits one BOS token and Gemma's `<|turn>` / `<|channel>` framing,
supports `enable_thinking`, and requires no runtime network access. CLI output
hides complete thought channels by default; `--show-thinking` preserves them.

## Fetch and run

The fetch helper prepares a fresh snapshot automatically without retaining a
second checkpoint-sized backup:

```bash
.venv/bin/python scripts/fetch_models.py \
  --tier 'gemma-4:31b-mlx' --target-dir models
export GEMMA4_31B_MLX_MODEL=$PWD/models/gemma-4-31b-mlx
edge0 demo 'gemma-4:31b-mlx'
```

Override the source with `GEMMA4_31B_MLX_REPO`. For an existing manual
download, backups are kept by default:

```bash
python scripts/prepare_gemma4_checkpoint.py /path/to/gemma-4-31b-4bit
```

Use `--no-backup` only for a snapshot that can be fetched again.

## Verification status

The non-slow suite covers a tiny six-layer model's prefill and cached decode,
cache kinds/offset continuity/reset, logit shape and finiteness, attention
shapes, sanitizer mapping/drop behavior, adapter/engine hooks, registry and
metadata, checkpoint preparation, chat formatting, and CLI thinking display.

A real-checkpoint smoke test is opt-in:

```bash
GEMMA4_31B_MLX_MODEL=/path/to/gemma-4-31b-mlx \
  pytest -m slow -q \
  tests/test_e2e_slow.py::test_gemma4_31b_mlx_real_checkpoint
```

That real-checkpoint test has not been run for this integration. Decode speed,
prefill throughput, and memory remain unmeasured; the acceptance metrics stay
at zero until measured.
