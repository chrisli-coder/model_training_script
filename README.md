# model_training_script

A **single-file**, nanoGPT-style language model trainer: **`train_gpt.py`** (~1k lines, internally split into logical sections). Author and version are defined at the top of the script as `__author__` and `__version__`, or run `python train_gpt.py --version`.

## Features

- **Configuration**: `TrainConfig` defaults → optional [`configs/default.yaml`](configs/default.yaml) → **CLI overrides win**; merged config is written to `out_dir/config_resolved.yaml` at startup.
- **Environment**: Staged checks before training (Python version, required packages, `device` vs CUDA/MPS, data paths, optional `tensorboard` / `wandb` / `tiktoken`).
- **Data**: supports raw UTF-8 text mode and pretokenized binary mode. Text mode reads a single `.txt` file or directory of `.txt` files; bin mode opens `train.bin` / `val.bin` with `np.memmap` so host memory stays roughly constant even for very large corpora.
- **Tokenizer**: **Character-level** by default; also supports `--tokenizer tiktoken` and external `tokenizer.json` files. Char tokenization in bin mode can reuse a saved vocab via `--vocab_file`.
- **Model**: Causal Transformer (nanoGPT-style blocks), `dropout` / `bias`, init, and parameter counts printed at startup.
- **Training**: `AdamW` by default, optional `bitsandbytes` **`AdamW8bit`** via `--optimizer_name adamw8bit`, **warmup + cosine** LR schedule down to `min_lr`, **gradient clipping**, configurable **gradient accumulation** via `--accumulation_steps`, optional **CUDA AMP**; **L2 gradient norm** (`grad_norm`) logged each eval.
- **Memory saving**: optional **gradient checkpointing** (`--gradient_checkpointing`) reduces activation memory during training by recomputing each Transformer block in backward; optional **8-bit optimizer** reduces optimizer-state memory on supported CUDA setups.
- **Evaluation & logging**: `val_loss` / perplexity on a validation split every `eval_interval`; `log_backend` is `none` | `tensorboard` | `wandb`.
- **Console progress line** (each eval): fixed-width `Step …/max_iters | percentage`, ASCII bar (e.g. `[====>---------------]`), `train_loss` / `val_loss` / `grad_norm` / `ppl` / `lr`; **`d_tr` / `d_va`** are loss deltas vs the **previous eval** (`n/a` on the first line; improvement is negative); **`s/it`** approximates seconds per training step from wall time between evals. TensorBoard/WandB also get `train/loss_delta`, `val/loss_delta`, `train/secs_per_iter`. The final **`Eval history`** block uses the same column layout **without** the progress bar to keep lines shorter.
- **Checkpoints**: `checkpoint_interval` → `latest.pt`; improved validation → `best.pt`; includes model, optimizer, scaler, RNG state (including NumPy); `torch.load(..., weights_only=False)` for full checkpoints on PyTorch 2.6+.
- **Resume**: `--resume` loads `out_dir/latest.pt` by default, or `--checkpoint path`.
- **Early stopping**: if `early_stop_patience > 0`, stop after that many evals without validation improvement.
- **Sampling**: periodic text samples at `sample_interval`; optional append to `out_dir/samples.txt`.
- **Summary**: after training, prints **`=== Eval history (every eval_interval) ===`**.
- **Lightweight help**: `python train_gpt.py -h` / `--version` exit **before** importing PyTorch (works without a full ML env).

## Requirements

- **Python 3.9+** (enforced in-script).
- See [`requirements.txt`](requirements.txt). Install **PyTorch** for your **CPU or CUDA** stack from the [official site](https://pytorch.org).

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# For GPU, follow PyTorch docs for the CUDA build you need
```

Optional extras:

```bash
pip install tiktoken tokenizers tensorboard wandb bitsandbytes
```

Notes:

- `bitsandbytes` `AdamW8bit` is typically for **CUDA** training. On unsupported devices like `cpu` or `mps`, the script can either fall back to standard `AdamW` or fail early, depending on `--optimizer_fallback`.
- Gradient checkpointing saves memory but usually increases compute time because activations are recomputed during backward.

## Quick start

1. Prepare either raw text data or pretokenized binary data.

   ```bash
   mkdir -p data
   # text mode: put *.txt under data/ or point --data_dir at one file
   # bin mode: create data/train.bin and data/val.bin with uint16/int32/int64 token ids
   ```

2. Run (example overrides):

   ```bash
   python train_gpt.py \
     --config configs/default.yaml \
     --data_dir data \
     --out_dir runs/exp01 \
     --device auto \
     --num_threads 8
   ```

3. For large corpora, switch to memmap-backed bin mode:

   ```bash
   python train_gpt.py \
     --data_dir data \
     --data_format bin \
     --tokenizer tokenizer.json \
     --batch_size 2 \
     --accumulation_steps 32 \
     --gradient_checkpointing \
     --amp
   ```

3. Help and version (**no PyTorch required**):

   ```bash
   python train_gpt.py --help
   python train_gpt.py --version
   ```

## Common CLI flags

| Area | Examples | Meaning |
|------|----------|---------|
| Device | `--device auto\|cpu\|cuda\|mps` | `auto`: CUDA → MPS → CPU |
| Training | `--batch_size` `--lr` `--max_iters` `--weight_decay` | Batch size, LR, total steps, weight decay |
| Effective batch | `--accumulation_steps` | Number of microbatches per optimizer update |
| Schedule | `--warmup_iters` `--min_lr` | Warmup steps and cosine floor LR |
| Model | `--n_layer` `--n_head` `--n_embd` `--block_size` | Depth, heads, width, context length |
| Data | `--data_dir` `--data_format text\|bin` `--train_bin` `--val_bin` `--token_dtype` | Raw text or memmap-backed token files |
| Tokenizer | `--tokenizer char\|tiktoken\|path/to/tokenizer.json` `--vocab_file` | Tokenizer backend and optional saved char vocab |
| I/O | `--out_dir` `--eval_interval` `--checkpoint_interval` | Output dir and log/checkpoint cadence |
| Resume | `--resume` `--checkpoint path/to.pt` | Continue from checkpoint |
| Logging | `--log_backend none\|tensorboard\|wandb` | Metrics backend |
| Optimizer | `--optimizer_name adamw\|adamw8bit` `--optimizer_fallback fallback\|strict` | Choose optimizer backend and unsupported-env behavior |
| Memory | `--gradient_checkpointing` | Recompute Transformer blocks in backward to reduce activation memory |
| Other | `--amp` `--early_stop_patience N` `--grad_clip` | AMP, early stop, clip norm |

The full list is in **`python train_gpt.py --help`**. YAML keys match `TrainConfig` field names (snake_case).

## Typical `out_dir` layout

| Path | Description |
|------|-------------|
| `config_resolved.yaml` | Final merged configuration |
| `startup_log.txt` | Startup report (same as stdout) |
| `latest.pt` / `best.pt` | Latest / best validation checkpoint |
| `samples.txt` | If sampling is on and `append_samples_to_file` |
| `tb/` | TensorBoard logs when `log_backend=tensorboard` |

The startup report now also shows the requested optimizer, resolved optimizer after fallback, optimizer warning text, and whether gradient checkpointing is active.

## Data Modes

### Text mode

Use the legacy workflow when corpora are small enough to tokenize in memory:

```bash
python train_gpt.py \
  --data_dir data \
  --data_format text \
  --tokenizer char
```

`train_ratio` and `val_ratio` are only used in this mode.

### Bin mode

Use this for large datasets or limited RAM. The script opens `train.bin` and `val.bin` lazily with `np.memmap`.

```bash
python train_gpt.py \
  --data_dir data \
  --data_format bin \
  --token_dtype uint16 \
  --tokenizer tokenizer.json
```

If you prefer explicit paths:

```bash
python train_gpt.py \
  --data_format bin \
  --train_bin /path/to/train.bin \
  --val_bin /path/to/val.bin \
  --tokenizer tiktoken
```

For char tokenization in bin mode, pass a saved vocab file:

```bash
python train_gpt.py \
  --data_format bin \
  --tokenizer char \
  --vocab_file char_vocab.txt
```

`char_vocab.txt` should contain the character set in token-id order. A JSON file with a top-level `chars` list is also accepted.

## Hardware-oriented examples

Laptop CUDA example with aggressive memory savings:

```bash
python train_gpt.py \
  --data_dir ./dataset \
  --data_format bin \
  --tokenizer ./tokenizer_v1/tokenizer.json \
  --n_layer 24 \
  --n_embd 2048 \
  --batch_size 2 \
  --accumulation_steps 32 \
  --gradient_checkpointing \
  --optimizer_name adamw8bit \
  --amp
```

CPU compatibility example:

```bash
python train_gpt.py \
  --device cpu \
  --data_dir ./dataset \
  --data_format text \
  --n_layer 6 \
  --n_embd 384 \
  --batch_size 16
```

## Repository layout

```
model_training_script/
  README.md
  train_gpt.py          # single entrypoint and implementation
  requirements.txt
  configs/default.yaml  # optional default hyperparameters
  plan.md               # design notes (if kept)
```

Artifacts (runs, `.venv`, etc.) should stay ignored via [`.gitignore`](.gitignore)—do not commit them.

## License & contributing

If no `LICENSE` file is present, all rights are reserved; add a license if you open-source the project.

---

Canonical **version** and **author** values live at the top of **`train_gpt.py`** (`__version__`, `__author__`).
