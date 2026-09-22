# Qwen3.8-27B MLX

`qwen3.8:27b-mlx` adds text-generation support for the dense [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) checkpoint. It uses the Qwen3.5-family hybrid Gated DeltaNet/full-attention implementation already vendored by edge0 and reuses the shared registry, CLI, server, cache, and generation loop.

Unlike the sparse edge0 tiers, this model has no routed experts. Its adapter therefore does not install `StreamingSwitchGLU`, a prerouter, or LoRA, and its quantized weights remain resident during inference. The checkpoint includes a vision tower, but the current edge0 adapter is text-only and discards vision weights while loading.

## Model profile

| Item | Value |
| --- | --- |
| Registered name | `qwen3.8:27b-mlx` |
| Hugging Face checkpoint | `mlx-community/Qwen3.8-27B-4bit` |
| Registry aliases | `qwen3_5`, `qwen3_5_text` |
| Architecture | Dense Qwen3.5-family hybrid attention |
| Layers | 64 |
| Hidden size | 5120 |
| Intermediate size | 17408 |
| Attention pattern | 48 Gated DeltaNet + 16 full-attention layers (interval 4) |
| Quantization | 4-bit affine, group 64 |
| Context limit | 262,144 tokens (memory permitting) |
| Default sampling | temperature 1.0, top-p 0.95, top-k 20 |
| EOS tokens | 248046, 248044 |
| Serving port | 8084 |
| Measured short-prompt throughput | 6.0 tok/s on Mac Studio M4 Max, 36 GB |
| Measured peak process footprint | 14.7 GiB |

The throughput figure is a 48-token smoke measurement including short-prompt prefill. It is not directly comparable with the long-prompt sparse-tier benchmark protocol.

## Download

```bash
python scripts/fetch_models.py \
  --tier 'qwen3.8:27b-mlx' \
  --target-dir models
```

The helper writes `models/qwen3.8-27b-mlx`. You can override the source repository with `QWEN38_27B_MLX_REPO`.

## Usage

Use the checkpoint path directly; registry resolution reads `model_type: qwen3_5` from `config.json`:

```bash
edge0 chat models/qwen3.8-27b-mlx \
  --prompt "Explain hybrid attention in one sentence."

edge0 serve models/qwen3.8-27b-mlx \
  --host 127.0.0.1 \
  --port 8084
```

Or configure the registered model name:

```bash
export QWEN38_27B_MLX_MODEL=$PWD/models/qwen3.8-27b-mlx
edge0 demo 'qwen3.8:27b-mlx'
```

Python API:

```python
from edge0 import AutoEngine

engine = AutoEngine.from_pretrained("models/qwen3.8-27b-mlx")
tokens = engine.generate([248044], max_new_tokens=32)
print(engine._tok.decode(tokens))
engine.close()
```

## Memory behavior

The 4-bit checkpoint occupies about 15 GB on disk. In a short generation test, macOS reported a 14.7 GiB peak process footprint. Long contexts increase cache memory substantially, so the checkpoint's 262K context limit should not be treated as a practical default on a 36 GB machine.
