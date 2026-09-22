# Muse-Glimmer-30B MLX

`muse-glimmer:30b-mlx` is Edge0's text-generation tier for
[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B).
The practical Apple-Silicon fetch default is
[`mlx-community/Muse-Glimmer-30B-4bit`](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit),
an MLX 4-bit conversion of that model.

## Current scope

Edge0 loads only the dense language model. The vision tower, vision adapter,
vision projection, perception modules, and their checkpoint tensors are
deliberately omitted. Image input is not yet wired through the Edge0 Python,
CLI, or server APIs; this tier currently accepts text only.

This is a resident dense model. It has no routed experts and never uses expert
shards, streaming installation, prerouting, LoRA, staging, history prefetch, or
hot-layer pins.

## Profile

| Setting | Value |
|---|---:|
| Transformer layers | 52 |
| Hidden size | 6656 |
| MLP intermediate size | 19968 |
| Attention heads / KV heads | 32 / 2 |
| Head dimension | 128 |
| Attention pattern | sliding, sliding, sliding, full (repeating) |
| Sliding window | 2048 tokens |
| Local-layer RoPE theta | 500000 |
| Full-layer RoPE | disabled |
| Quantization | 4-bit affine, group size 64 |
| Server port | 8086 |

The text implementation preserves Muse Glimmer's centered RMS norms,
attention gate, `qk_scale_factor=3.87`, output multiplier
`0.19611613513818404`, and final logit softcap `20.0`. It is adapted from the
MIT-licensed mlx-vlm 0.6.12 implementation and uses APIs already present in
Edge0's pinned `mlx==0.30.6` / `mlx-lm==0.31.0`. `mlx-vlm` is deliberately
not a dependency because that release requires `mlx>=0.32` and
`mlx-lm>=0.31.3`, which conflicts with Edge0's runtime pins.

Checkpoint sanitization retains already-normalized `language_model.*` keys,
maps official `model.language_model.*` into `language_model.model.*`, maps
top-level `lm_head.*`, and drops all known vision/perception tensors before
strict loading.

## Generation defaults

The checkpoint declares `do_sample=false`, so the tier defaults are
deterministic: temperature `0.0`, top-p `1.0`, top-k `0`, repetition penalty
`1.0`, EOS IDs `200001` and `200008`, and at most 2048 new tokens.

## Fetch and run

No weights are bundled or downloaded during installation. Fetch explicitly:

```bash
.venv/bin/python scripts/fetch_models.py \
  --tier 'muse-glimmer:30b-mlx' --target-dir models
export MUSE_GLIMMER_30B_MODEL=$PWD/models/muse-glimmer-30b-mlx
edge0 chat muse-glimmer:30b-mlx \
  --prompt "Write one short sentence about light shimmering on water."
```

Override the fetch source with `MUSE_GLIMMER_30B_REPO`.

## Verification status

The non-slow suite includes tiny-model prefill and cached single-token decode,
cache continuity/reset, sanitizer, adapter, registry, and CLI/fetch tests. A
real-checkpoint smoke test is available when `MUSE_GLIMMER_30B_MODEL` is set:

```bash
MUSE_GLIMMER_30B_MODEL=/path/to/muse-glimmer-30b-mlx \
  pytest -m slow -q \
  tests/test_e2e_slow.py::test_muse_glimmer_30b_mlx_real_checkpoint
```

No Edge0 decode-speed, prefill-throughput, or memory figures have been measured
for this tier yet. Do not infer them from other runtimes. Full-checkpoint Edge0
runtime verification should be recorded here only after the gated smoke test is
actually run against the downloaded conversion.
