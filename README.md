# model_training_script

A **single-file**, nanoGPT-style language model trainer: **`train_gpt.py`** (~1k lines, internally split into logical sections). Author and version are defined at the top of the script as `__author__` and `__version__`, or run `python train_gpt.py --version`.

## Features

- **Configuration**: `TrainConfig` defaults → optional [`configs/default.yaml`](configs/default.yaml) → **CLI overrides win**; merged config is written to `out_dir/config_resolved.yaml` at startup.
- **Environment**: Staged checks before training (Python version, required packages, `device` vs CUDA/MPS, data paths, optional `tensorboard` / `wandb` / `tiktoken`).
- **Data**: `data_dir` is either a **single `.txt` file** or a **directory of `.txt` files** (UTF-8); sequence is split by `train_ratio` / `val_ratio`.
- **Tokenizer**: **Character-level** by default; `--tokenizer tiktoken` requires `tiktoken` (default `tiktoken_encoding` e.g. `gpt2`).
- **Model**: Causal Transformer (nanoGPT-style blocks), `dropout` / `bias`, init, and parameter counts printed at startup.
- **Training**: AdamW, **warmup + cosine** LR schedule down to `min_lr`, **gradient clipping**, optional **CUDA AMP**; **L2 gradient norm** (`grad_norm`) logged each eval.
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
pip install tiktoken tensorboard wandb
```

## Quick start

1. Prepare text data—a folder of `.txt` files or one file:

   ```bash
   mkdir -p data
   # put *.txt under data/ or point --data_dir at one file
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
| Schedule | `--warmup_iters` `--min_lr` | Warmup steps and cosine floor LR |
| Model | `--n_layer` `--n_head` `--n_embd` `--block_size` | Depth, heads, width, context length |
| Data | `--data_dir` `--train_ratio` `--val_ratio` | Path and split ratios |
| I/O | `--out_dir` `--eval_interval` `--checkpoint_interval` | Output dir and log/checkpoint cadence |
| Resume | `--resume` `--checkpoint path/to.pt` | Continue from checkpoint |
| Logging | `--log_backend none\|tensorboard\|wandb` | Metrics backend |
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
