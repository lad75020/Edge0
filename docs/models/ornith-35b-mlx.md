# Ornith-1.0-35B MLX

`ornith:35b-mlx` is Edge0's text-generation adapter for
[`mlx-community/Ornith-1.0-35B-4bit`](https://huggingface.co/mlx-community/Ornith-1.0-35B-4bit),
an MIT-licensed MLX 4-bit conversion of
`deepreinforce-ai/Ornith-1.0-35B`. The verified source revision is
`781f91090809411b7fc07449817f398a99feb188`, converted with mlx-vlm 0.6.3.
Edge0 does not add an mlx-vlm runtime dependency.

## Scope: text only

The upstream checkpoint is a VLM. Edge0 supports only its language model:

- `scripts/prepare_ornith_checkpoint.py` removes all 333
  `vision_tower.*` tensors while retaining every indexed language tensor;
- the safetensors index and shards are rewritten coherently and
  `metadata.total_size` is recomputed;
- image and video inputs are not wired through Edge0 APIs.

The verified upstream index contains 2,090 tensors: 1,757 language tensors
and 333 vision tensors, with 20,401,929,952 total tensor bytes before
preparation.

## Architecture and execution profile

- vendored implementation: `edge0.backends.mlx._impl.qwen3_5_moe`;
- engine loader: `edge0.engine.qwen.load_installed`;
- 40 layers, hidden size 2,048;
- layer pattern: `linear, linear, linear, full`;
- head dimension 256, 16 query heads, 2 key/value heads;
- RoPE theta 10,000,000;
- 256 routed experts, top-k 8, routed intermediate size 512;
- one shared expert with intermediate size 512;
- softmax top-k routing with `norm_topk_prob`;
- fused `switch_mlp` checkpoint keys already match Edge0's streaming layout;
- expert weights use 4-bit affine group-64 quantization; router and shared
  expert gates remain 8-bit as encoded by the checkpoint.

Decode uses exact current-token gate routing. There is no prerouter, LoRA,
history-filled staging, previous-token staged replacement, hot window,
intra-layer staging, or history prefetch. Prefill uses the full-layer
load/drop path for all layers.

## Download and prepare

The fetch helper downloads directly into `models/ornith-35b-mlx` and prepares
the fresh snapshot automatically without retaining a second checkpoint-sized
backup:

```bash
python scripts/fetch_models.py --tier 'ornith:35b-mlx' --target-dir models
export ORNITH_35B_MLX_MODEL=$PWD/models/ornith-35b-mlx
edge0 demo 'ornith:35b-mlx'
```

For an existing manual download, preparation keeps backups by default:

```bash
python scripts/prepare_ornith_checkpoint.py /path/to/Ornith-1.0-35B-4bit
```

Use `--no-backup` only when the source can be downloaded again. Rewrites are
staged in same-directory temporary files and installed with atomic file
replacement.

Preparation preserves the upstream top-level `model_type=qwen3_5_moe` and
`text_config.model_type=qwen3_5_moe_text`, then adds the top-level marker:

```json
{
  "edge0_model_name": "ornith:35b-mlx",
  "model_type": "qwen3_5_moe"
}
```

The marker lets path auto-detection select Ornith without remapping the
generic `qwen3_5_moe` alias away from `edge0-35b`.

## Generation and measurement status

Checkpoint defaults are temperature 1.0, top-p 0.95, top-k 20, repetition
penalty 1.0, EOS IDs `(248046, 248044)`, and 2,048 maximum new tokens. The
default server port is 8087.

Decode throughput and peak active memory are intentionally recorded as zero
in the acceptance profile until measured. No performance or memory claim is
made for this tier yet.

The real-checkpoint smoke test is opt-in:

```bash
ORNITH_35B_MLX_MODEL=/path/to/ornith-35b-mlx \
  pytest -m slow -q tests/test_e2e_slow.py::test_ornith_35b_mlx_real_checkpoint
```
