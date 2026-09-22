<div align="center">

<img src="assets/20260908-223115.jpg" alt="edge0" width="100%">

# edge0

**开源流式 MoE 推理框架 —— SSD 专家 offload + Recover-LoRA + prerouter 路由预判**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--35B--A3B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Edge0--8B--A1B--preview-yellow?style=for-the-badge)](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)
[![GitHub](https://img.shields.io/badge/GitHub-Edge0--AI%2FEdge0-black?style=for-the-badge&logo=github)](https://github.com/Edge0-AI/Edge0)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

[English](README.md) | 中文

</div>

**edge0** 是一个开源的流式 MoE 推理框架：把「SSD 专家 offload +
Recover-LoRA + prerouter 路由预判」抽象成可扩展的通用框架。后端隔离设计，
当前实现 MLX 后端（Apple Silicon），更多平台（CUDA 等）即将接入。

框架随附两个端到端稀疏模型档位，并支持常驻内存的 Qwen3.8、仅文本
Gemma 4 与 Muse Glimmer，以及精确门控路由的 Ornith MLX checkpoint。

| 档位 | 发布 checkpoint | 推理档 |
|---|---|---|
| `edge0-35b` | [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) | 4bit，40 层，256 专家，prerouter K=4 |
| `edge0-8b` | [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview) | 4bit，24 层，128 专家，prerouter K=8 |
| `gemma-4:31b-mlx` | [`mlx-community/gemma-4-31b-4bit`](https://huggingface.co/mlx-community/gemma-4-31b-4bit) | 4bit 稠密模型，60 层常驻文本架构 |
| `muse-glimmer:30b-mlx` | [`mlx-community/Muse-Glimmer-30B-4bit`](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit) | 4bit 稠密模型，52 层文本架构，MLX 常驻推理 |
| `ornith:35b-mlx` | [`mlx-community/Ornith-1.0-35B-4bit`](https://huggingface.co/mlx-community/Ornith-1.0-35B-4bit) | 4bit 稀疏模型，40 层、256 专家、精确门控 K=8；仅文本 |
| `qwen3.8:27b-mlx` | [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 4bit 稠密模型，64 层，MLX 常驻推理 |

Muse checkpoint 是 `meta-models/Muse-Glimmer-30B` 的 MLX 4bit 转换。
Edge0 目前只加载文本架构：会丢弃视觉张量，图片输入尚未接入 Edge0 API。

Gemma checkpoint 是 `google/gemma-4-31b` 的 MLX 4bit 转换。Edge0 的确定性
准备步骤会移除 355 个 `vision_tower.*` 与 3 个 `embed_vision.*` 张量，安装
本地规范文本对话模板，并且仅支持文本生成；图片、音频和视频输入均未接入。
模型权重仍受 Gemma 条款约束，改编的 mlx-vlm 模型代码采用 MIT 许可证。

Ornith 同样以 VLM checkpoint 发布。Edge0 的准备步骤会移除其中 333 个
视觉张量，仅支持文本生成；图片和视频输入尚未接入，也不依赖 mlx-vlm。

两个 Edge0 发布 checkpoint 均基于开源稀疏 MoE 基座（分别为 Qwen3.6-35B-A3B
与 Ling 3.0 混合架构），并携带为本框架训练的 LoRA 与 prerouter 权重——
适配器文件与 checkpoint 同目录、自动加载，`edge0 serve <tier>` 开箱即跑
训练好的完整管线。

## 环境要求

- **系统 / 硬件**：MLX 后端目前仅支持 Apple Silicon 的 macOS
  （M1/M2/M3/M4）；CUDA 后端在路线图中，其余平台暂不支持。
- **Python**：3.10+（推荐 3.12）。
- **MLX**：`mlx==0.30.6` / `mlx-metal==0.30.6`（`mlx-lm==0.31.0`，见
  `pyproject.toml`）。Apple A18 / A18 Pro 上输出乱码 = mlx 版本旧：
  `pip install 'mlx==0.30.6' 'mlx-metal==0.30.6'`
  （[#8](https://github.com/Edge0-AI/Edge0/issues/8)）。
- **内存**：短上下文下 `edge0-35b` ≈2.9 GB、`edge0-8b` ≈1.0 GB
  峰值激活内存；`qwen3.8:27b-mlx` 的峰值进程占用约 14.7 GiB
  （见[性能实测](#性能实测)）。Gemma 4、Muse Glimmer 与 Ornith 在 Edge0 中尚无内存实测。
  另为系统、tokenizer 与长上下文 KV 增长预留余量。
- **磁盘**：4bit checkpoint 约 23 GB（`edge0-35b`）/ 4.2 GB
  （`edge0-8b`），Qwen3.8 MLX checkpoint 约 15 GB；Ornith 源张量在移除
  视觉权重前约 20.4 GB。稀疏专家权重 mmap 按需读取，Qwen3.8 稠密权重常驻内存。

## 设计

- **像 transformers 一样使用**：`AutoModel` / `AutoConfig` / `AutoEngine`
  按模型名自动选类；
- **后端隔离**：全部 MLX 代码收在 `edge0/backends/mlx/`，核心逻辑
  （模型 spec / prerouter / 流式专家池 / server）只依赖后端门面
  （`edge0/backends/base.py` 的 `core` / `nn` 门面），新增后端实现同一门面
  即可平级接入（`backends/cuda/` 预留插槽），核心代码零改动；
- **适配器统一为 safetensors**：LoRA 与 prerouter 权重均为带元数据
  （来源、版本、owner 层）的 `.safetensors`，放模型目录或 `artifacts/`
  均可自动解析；
- **模型 + 适配器同目录布局**：一个模型目录同时放基模（`config.json` /
  `model*.safetensors` / tokenizer）和该模型的适配器，升级适配器只换适配器文件
  文件，基模不动、不 merge。

## 核心机制

- **SSD 专家 offload**：专家权重按需从存储流式加载，峰值内存由
  激活集而非参数量决定；
- **prerouter**：训练头提前一步预测专家路由，专家装载与前向计算重叠
  而非阻塞——解码吞吐**最高 +59%**，收益随存储延迟、模型规模与路由
  宽度 *K* 增大；
- **Recover-LoRA**：冻结 int4 基模，用 FP teacher 蒸馏训练 LoRA，
  在 4bit 下恢复绝大部分量化损失（见[质量](#质量)）。适配器不合并，
  一份只读基模服务多套适配器。

## 快速开始

### 1) 安装

```bash
# Python ≥3.10；MLX 后端需 macOS + Apple Silicon
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev,fetch]'
```

### 2) 下载模型

所有支持的 checkpoint 均发布在 Hugging Face。两个 Edge0 发布档位将 LoRA +
prerouter 与基模放在同一目录；Qwen3.8、Gemma 4 与 Muse Glimmer 是稠密 MLX checkpoint，
Ornith 则是准备为仅文本流式推理的稀疏 VLM checkpoint：

- [`Edge0/Edge0-35B-A3B-preview`](https://huggingface.co/Edge0/Edge0-35B-A3B-preview)（约 23 GB）
- [`Edge0/Edge0-8B-A1B-preview`](https://huggingface.co/Edge0/Edge0-8B-A1B-preview)（约 4.2 GB）
- [`mlx-community/gemma-4-31b-4bit`](https://huggingface.co/mlx-community/gemma-4-31b-4bit)（`google/gemma-4-31b` 的 MLX 4bit 转换；准备后仅文本）
- [`mlx-community/Muse-Glimmer-30B-4bit`](https://huggingface.co/mlx-community/Muse-Glimmer-30B-4bit)（`meta-models/Muse-Glimmer-30B` 的 MLX 4bit 转换）
- [`mlx-community/Ornith-1.0-35B-4bit`](https://huggingface.co/mlx-community/Ornith-1.0-35B-4bit)（准备后在 Edge0 中仅支持文本）
- [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)（约 15 GB）

```bash
# 用仓库自带脚本（默认即上述六个仓库）：
.venv/bin/python scripts/fetch_models.py --tier edge0-35b --target-dir models
.venv/bin/python scripts/fetch_models.py --tier edge0-8b --target-dir models
.venv/bin/python scripts/fetch_models.py --tier 'gemma-4:31b-mlx' --target-dir models
.venv/bin/python scripts/fetch_models.py --tier 'muse-glimmer:30b-mlx' --target-dir models
.venv/bin/python scripts/fetch_models.py --tier 'ornith:35b-mlx' --target-dir models
.venv/bin/python scripts/fetch_models.py --tier 'qwen3.8:27b-mlx' --target-dir models

# 或直接用 CLI：
.venv/bin/huggingface-cli download Edge0/Edge0-35B-A3B-preview \
    --local-dir models/edge0-35b
.venv/bin/huggingface-cli download Edge0/Edge0-8B-A1B-preview \
    --local-dir models/edge0-8b
.venv/bin/huggingface-cli download mlx-community/gemma-4-31b-4bit \
    --local-dir models/gemma-4-31b-mlx
.venv/bin/python scripts/prepare_gemma4_checkpoint.py models/gemma-4-31b-mlx --no-backup
.venv/bin/huggingface-cli download mlx-community/Muse-Glimmer-30B-4bit \
    --local-dir models/muse-glimmer-30b-mlx
.venv/bin/huggingface-cli download mlx-community/Ornith-1.0-35B-4bit \
    --local-dir models/ornith-35b-mlx
.venv/bin/python scripts/prepare_ornith_checkpoint.py models/ornith-35b-mlx --no-backup
.venv/bin/huggingface-cli download mlx-community/Qwen3.8-27B-4bit \
    --local-dir models/qwen3.8-27b-mlx
```

下载完成后目录结构：

```
models/edge0-35b/
├── config.json, model-*.safetensors, tokenizer 文件   # 基模 checkpoint
├── lora_edge0_35b.safetensors          # 训练好的 LoRA 适配器
└── prerouter_edge0_35b.safetensors     # 训练好的 prerouter 头
```

### 3) 指向模型目录

档位名经环境变量解析到本地目录（放哪由你决定）：

```bash
export EDGE0_35B_MODEL=$PWD/models/edge0-35b
export EDGE0_8B_MODEL=$PWD/models/edge0-8b
export GEMMA4_31B_MLX_MODEL=$PWD/models/gemma-4-31b-mlx
export MUSE_GLIMMER_30B_MODEL=$PWD/models/muse-glimmer-30b-mlx
export ORNITH_35B_MLX_MODEL=$PWD/models/ornith-35b-mlx
export QWEN38_27B_MLX_MODEL=$PWD/models/qwen3.8-27b-mlx
```

也可以不用环境变量，直接传目录——框架从 checkpoint 的 `config.json`
自动识别档位：

```bash
edge0 demo models/edge0-35b
edge0 serve models/edge0-8b
edge0 chat models/gemma-4-31b-mlx --prompt "用一句话解释部分旋转注意力。"
edge0 chat models/muse-glimmer-30b-mlx --prompt "用一句话描写水面微光。"
edge0 chat models/ornith-35b-mlx --prompt "用一句话解释强化学习。"
edge0 chat models/qwen3.8-27b-mlx --prompt "用一句话解释混合注意力。"
```

### 4) 运行

```bash
# 快速演示
edge0 demo edge0-35b

# 起服务（OpenAI 兼容 /v1/chat/completions）
edge0 serve edge0-35b
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":32}'

# 5) 单轮对话（--max-new 限制生成长度；加 --show-thinking 会一并打印思考块）
edge0 chat edge0-35b --prompt "用一句话解释流式推理。"
```

`python -m edge0 ...` 等价于 `edge0 ...`。

### Python API

```python
from edge0 import AutoEngine
from edge0.server.chat import ChatMessage, ChatRequest, ChatSession

engine = AutoEngine.from_pretrained("/path/to/model")  # tier 自动识别
req = ChatRequest(
    model=engine.name,
    messages=[ChatMessage(role="user", content="你好！")],
    max_tokens=64,
)
tokens, meta = ChatSession(engine, req).run()
print(engine._tok.decode(tokens))
engine.close()   # 释放 mmap / 专家缓存
```

`examples/demo.py` 就是这条最小路径（`edge0 demo` 内部等价运行）。

### 模型与适配器

- **checkpoint**：原始模型目录（`config.json`、`model*.safetensors`、
  tokenizer）。`edge0 serve <dir>` / `AutoEngine.from_pretrained(<dir>)`
  按 `config.json` 自动识别 tier。
- **适配器**（LoRA + prerouter，safetensors）放两处任一，自动解析：
  - **模型目录内**（推荐）：与基模同目录，如
    `lora_edge0_35b.safetensors` + `prerouter_edge0_35b.safetensors`；
  - `artifacts/`（仓库根，gitignored）：`edge0 convert-adapters` 从
    训练侧 npz 一次性转换。
- 发布的模型仓库同时包含基模与当前默认适配器版本，
  `scripts/fetch_models.py` 下载后即为可运行的模型目录。适配器来源
  （训练数据、owner 层分布）见各模型文档页。
- prerouter + LoRA 两条适配器都必需；缺文件时 `edge0` 会给出明确报错
  （也可加 `--no-prerouter` / `--no-lora` 直接跑裸基模）。

## 质量

全部评测由我们使用 [OpenCompass](https://github.com/open-compass/opencompass)、
在完全相同的设置与参数下对 edge0 模型（int4 + 训练适配器 + prerouter 路由）
与原 fp16 基座模型测得。edge0 管线的损失很小：**edge0-35b 平均仅落后
3.9 分、edge0-8b 落后 2.8 分**（MMLU-Pro 甚至反超基座）。满分 100：

| 评测集 | edge0-35b（int4） | Qwen3.6-35B-A3B（fp16） | edge0-8b（int4） | Ling 3.0 tiny（fp16） |
|---|---:|---:|---:|---:|
| AIME 2026 | 86.6 | 92.7 | 63.3 | 73.3 |
| HumanEval | 90.9 | 95.1 | 91.5 | 92.7 |
| GPQA-Diamond | 79.8 | 81.8 | 70.7 | 71.2 |
| MMLU-Pro | 81.0 | 84.6 | 70.1 | 65.8 |
| IFBench | 57.9 | 61.7 | 53.9 | 60.6 |
| **平均** | **79.2** | **83.2** | **69.9** | **72.7** |

## 性能实测

`examples/bench.py` 实测（3.3k token prompt prefill → 10 步采样 warmup →
200 token 计时段，每档 2 轮）：

| 档位 | 解码速度 | Prefill 吞吐（冷/热）* | 峰值 active 内存 | 测试机器 |
|---|---|---|---|---|
| `edge0-35b` | 14.9–17.7 tok/s | 113 / 140 tok/s | 2.9 GiB | Mac mini M4 Pro, 24 GB |
| `edge0-8b` | 23.9–25.3 tok/s | 500 / 1428 tok/s | 1.0 GiB | Mac mini M4 Pro, 24 GB |
| `gemma-4:31b-mlx` | 尚未实测 | 尚未实测 | 尚未实测 | — |
| `muse-glimmer:30b-mlx` | 尚未实测 | 尚未实测 | 尚未实测 | — |
| `ornith:35b-mlx` | 尚未实测 | 尚未实测 | 尚未实测 | — |
| `qwen3.8:27b-mlx` | 6.0 tok/s† | — | 14.7 GiB 进程占用 | Mac Studio M4 Max, 36 GB |

*冷 = 进程启动后首请求（专家权重从 SSD 逐页换入）；热 = 后续请求（页缓存常驻）。Prefill 为 ≈3.3k token 长 prompt 的吞吐（`BENCH_LONG=1`）。

†Qwen3.8 数据来自含短 prompt prefill 的 48-token 冒烟测试，与稀疏档位的长 prompt benchmark 不完全可比。

复现：

```bash
python examples/bench.py edge0-35b    # 经 $EDGE0_35B_MODEL
python examples/bench.py edge0-8b    # 经 $EDGE0_8B_MODEL
edge0 chat qwen3.8:27b-mlx --prompt "简要解释混合注意力。"
```

## 验证

```bash
pytest                 # 单元测试（不含真实权重）
EDGE0_8B_MODEL=/path/to/edge0-8b pytest -m slow -q
                        # 真实权重生成测试；缺少的档位会明确 skip
QWEN38_27B_MLX_MODEL=/path/to/qwen3.8-27b-mlx \
  pytest -m slow -q tests/test_e2e_slow.py::test_qwen38_27b_mlx_real_checkpoint
MUSE_GLIMMER_30B_MODEL=/path/to/muse-glimmer-30b-mlx \
  pytest -m slow -q tests/test_e2e_slow.py::test_muse_glimmer_30b_mlx_real_checkpoint
ORNITH_35B_MLX_MODEL=/path/to/ornith-35b-mlx \
  pytest -m slow -q tests/test_e2e_slow.py::test_ornith_35b_mlx_real_checkpoint
GEMMA4_31B_MLX_MODEL=/path/to/gemma-4-31b-mlx \
  pytest -m slow -q tests/test_e2e_slow.py::test_gemma4_31b_mlx_real_checkpoint
.venv/bin/python scripts/e2e_smoke.py \
  --qwen-dir /path/to/edge0-35b --ling-dir /path/to/edge0-8b
                        # staged vs exact 一致性 + 生成冒烟
scripts/generate_example.py   # 完整 API 上手例子（prefill→生成→解码全链路）
examples/demo.py       # 最小 API walkthrough（edge0 demo 的等价代码）
```

## 文档

- [架构总览](docs/architecture.md)
- [注意力抽象](docs/attention.md) / [MoE 抽象](docs/moe.md) / [SSD 流式](docs/streaming.md) / [prerouter](docs/prerouter.md)
- [如何接入新模型](docs/adding-a-model.md)
- [edge0-35b](docs/models/edge0-35b.md) / [edge0-8b](docs/models/edge0-8b.md) / [Gemma 4 31B MLX](docs/models/gemma-4-31b-mlx.md) / [Muse-Glimmer-30B MLX](docs/models/muse-glimmer-30b-mlx.md) / [Ornith-1.0-35B MLX](docs/models/ornith-35b-mlx.md) / [Qwen3.8-27B MLX](docs/models/qwen3.8-27b-mlx.md)

## License

Edge0 采用 Apache-2.0；改编的第三方代码保留各自许可证（详见
[NOTICE](NOTICE)）。下载的 Gemma 模型权重不属于 Edge0，仍受 Gemma 条款约束。
