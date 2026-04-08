# model_training_script

单文件 **nanoGPT 风格**语言模型训练脚本：**`train_gpt.py`**（约千行，内部分区模块化）。作者与版本见脚本内 `__author__` / `__version__`，或通过 `--version` 查看。

## 功能概览

- **配置**：`TrainConfig` 默认值 → 可选 [`configs/default.yaml`](configs/default.yaml) → **命令行优先**；启动时写入 `out_dir/config_resolved.yaml`。
- **环境**：启动前分阶段检查（Python、依赖包、`device` 与 CUDA/MPS、数据路径、可选 `tensorboard` / `wandb` / `tiktoken`）。
- **数据**：`data_dir` 为**单个 `.txt` 文件**或**包含若干 `.txt` 的目录**（UTF-8）；按 `train_ratio` / `val_ratio` 切分序列。
- **Tokenizer**：默认**字符级**；`--tokenizer tiktoken` 时需安装 `tiktoken`（`tiktoken_encoding` 默认可为 `gpt2`）。
- **模型**：因果 Transformer（与常见 nanoGPT 结构一致），支持 `dropout`、`bias`、权重初始化与参数量打印。
- **训练**：AdamW、**warmup + 余弦**学习率至 `min_lr`、**梯度裁剪**、可选 **CUDA AMP**；每步可记录 **梯度 L2 范数**（`grad_norm`）。
- **评估与日志**：按 `eval_interval` 在验证集上算 `val_loss` / perplexity；`log_backend` 可选 `none` | `tensorboard` | `wandb`。
- **断点**：`checkpoint_interval` 写 `latest.pt`，验证改善写 `best.pt`；含 model / optimizer / scaler / RNG 等（含 NumPy RNG，加载时使用 `weights_only=False` 以兼容 PyTorch 2.6+）。
- **续训**：`--resume`，默认从 `out_dir/latest.pt` 恢复，或用 `--checkpoint` 指定路径。
- **早停**：`early_stop_patience > 0` 时，连续若干次验证未提升则结束。
- **生成**：`sample_interval` 间隔下采样输出；可选写入 `out_dir/samples.txt`。
- **收尾输出**：训练结束后打印 **`=== Eval history (every eval_interval) ===`** 汇总表。
- **无重依赖时也可用**：`python train_gpt.py -h` / `--version` 在导入 PyTorch 之前即可退出（便于未装环境时查看帮助）。

## 环境要求

- **Python 3.9+**（脚本内环境检查以 3.9 为下限）。
- 依赖见 [`requirements.txt`](requirements.txt)；**PyTorch** 请按 [官网](https://pytorch.org) 选择与本机 **CPU / CUDA** 匹配的轮子。

### 安装示例

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 若需 GPU，请按 PyTorch 文档选择带 CUDA 的安装命令
```

可选：

```bash
pip install tiktoken tensorboard wandb
```

## 快速开始

1. 准备语料，例如目录 `data/` 下多个 `.txt`，或单个文件：

   ```bash
   mkdir -p data
   # 将文本放入 data/*.txt 或指定 file.txt
   ```

2. 运行（示例覆盖部分超参）：

   ```bash
   python train_gpt.py \
     --config configs/default.yaml \
     --data_dir data \
     --out_dir runs/exp01 \
     --device auto \
     --num_threads 8
   ```

3. 查看帮助与版本（**无需**已安装 PyTorch 即可执行）：

   ```bash
   python train_gpt.py --help
   python train_gpt.py --version
   ```

## 常用参数说明

| 类别 | 参数示例 | 含义 |
|------|-----------|------|
| 设备 | `--device auto\|cpu\|cuda\|mps` | `auto`：CUDA → MPS → CPU |
| 训练 | `--batch_size` `--lr` `--max_iters` `--weight_decay` | 批大小、学习率、总步数、权重衰减 |
| 调度 | `--warmup_iters` `--min_lr` | 预热与余弦终点学习率 |
| 模型 | `--n_layer` `--n_head` `--n_embd` `--block_size` | 层数、头数、宽度、上下文长度 |
| 数据 | `--data_dir` `--train_ratio` `--val_ratio` | 路径与序列切分比例 |
| I/O | `--out_dir` `--eval_interval` `--checkpoint_interval` | 输出目录与评估/存盘频率 |
| 续训 | `--resume` `--checkpoint path/to.pt` | 从检查点继续 |
| 监控 | `--log_backend none\|tensorboard\|wandb` | 日志后端 |
| 其它 | `--amp` `--early_stop_patience N` `--grad_clip` | 混合精度、早停、裁剪阈值 |

完整列表以 **`python train_gpt.py --help`** 为准；YAML 中的键名与 CLI 长选项对应（蛇形命名与字段名一致）。

## 输出目录（`out_dir`）典型文件

| 文件 | 说明 |
|------|------|
| `config_resolved.yaml` | 合并后的最终配置 |
| `startup_log.txt` | 启动阶段完整报告（与控制台一致） |
| `latest.pt` / `best.pt` | 最近一次 / 验证最优检查点 |
| `samples.txt` | 若开启间隔采样且 `append_samples_to_file` |
| `tb/` | `log_backend=tensorboard` 时的日志 |

## 项目结构（仓库内）

```
model_training_script/
  README.md
  train_gpt.py          # 唯一训练入口与实现
  requirements.txt
  configs/default.yaml  # 可选默认超参
  plan.md               # 设计/计划说明（若保留）
```

训练产物、虚拟环境等应由 [`.gitignore`](.gitignore) 排除，勿提交。

## 许可证与贡献

未随仓库附带许可证文件时，默认保留所有权利；若需开源请自行添加 `LICENSE`。

---

版本与作者以 **`train_gpt.py` 顶部** `__version__`、`__author__` 为准。
