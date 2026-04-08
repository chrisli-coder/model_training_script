#!/usr/bin/env python3
# Single-file modular GPT trainer: YAML + CLI, checkpoints, optional TB/WandB.

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Chris Li"

import argparse
import dataclasses
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union


@dataclass
class TrainConfig:
    device: str = "auto"
    num_threads: int = 0
    seed: int = 1337
    batch_size: int = 64
    lr: float = 3e-4
    max_iters: int = 5000
    weight_decay: float = 0.1
    warmup_iters: int = 200
    min_lr: float = 1e-5
    grad_clip: float = 1.0
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = True
    data_dir: str = "data"
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    tokenizer: str = "char"
    tiktoken_encoding: str = "gpt2"
    out_dir: str = "runs/out"
    eval_interval: int = 500
    checkpoint_interval: int = 500
    resume: bool = False
    checkpoint: str = ""
    early_stop_patience: int = 0
    sample_interval: int = 500
    sample_prompt: str = ""
    sample_max_new_tokens: int = 200
    append_samples_to_file: bool = True
    log_backend: str = "none"
    num_workers: int = 0
    pin_memory: bool = True
    amp: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95


def _coerce_field_value(raw: Any, hint: type) -> Any:
    if raw is None:
        return None
    if hint is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(raw)
    if hint is int:
        return int(raw)
    if hint is float:
        return float(raw)
    if hint is str:
        return str(raw)
    return raw


def parse_cli() -> argparse.Namespace:
    """CLI only (stdlib). Safe before numpy/torch/yaml so `-h` / `--version` work with minimal deps."""
    p = argparse.ArgumentParser(description=f"Single-file GPT trainer (v{__version__}, {__author__})")
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} (author: {__author__})",
    )
    p.add_argument("--config", type=str, default="", help="YAML config path")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--warmup_iters", type=int, default=None)
    p.add_argument("--min_lr", type=float, default=None)
    p.add_argument("--grad_clip", type=float, default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--bias", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--train_ratio", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--tokenizer", type=str, default=None)
    p.add_argument("--tiktoken_encoding", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--checkpoint_interval", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--early_stop_patience", type=int, default=None)
    p.add_argument("--sample_interval", type=int, default=None)
    p.add_argument("--sample_prompt", type=str, default=None)
    p.add_argument("--sample_max_new_tokens", type=int, default=None)
    p.add_argument("--append_samples_to_file", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--log_backend", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--pin_memory", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None)
    p.add_argument("--amp", action="store_true", default=False)
    return p.parse_args()


# Help / version before heavy imports (no numpy/torch required).
if __name__ == "__main__" and ("-h" in sys.argv or "--help" in sys.argv or "--version" in sys.argv):
    parse_cli()

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml


def merge_yaml_into_config(cfg: TrainConfig, yaml_path: Optional[str]) -> TrainConfig:
    if not yaml_path:
        return cfg
    p = Path(yaml_path).expanduser()
    if not p.is_file():
        print(f"[fatal] Config file not found: {p}", file=sys.stderr)
        sys.exit(1)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        print(f"[fatal] YAML root must be a mapping: {p}", file=sys.stderr)
        sys.exit(1)
    merged = asdict(cfg)
    for k, v in data.items():
        if k in merged:
            merged[k] = _coerce_field_value(v, type(getattr(cfg, k)))
    return TrainConfig(**merged)


def check_environment_phase_a() -> dict[str, Any]:
    info: dict[str, Any] = {}
    if sys.version_info < (3, 9):
        print("[fatal] Python 3.9+ required. Got:", sys.version.split()[0], file=sys.stderr)
        sys.exit(1)
    try:
        import tqdm  # noqa: F401
    except ImportError:
        print("[fatal] Missing: tqdm\n  pip install tqdm", file=sys.stderr)
        sys.exit(1)
    try:
        info["torch_version"] = torch.__version__
    except Exception as e:
        print(f"[fatal] torch: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        import yaml as _yaml  # noqa: F401
    except ImportError:
        print("[fatal] Missing: PyYAML\n  pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    info["cuda_available"] = torch.cuda.is_available()
    info["mps_available"] = bool(
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    )
    return info


def resolve_device(device_str: str) -> torch.device:
    d = device_str.lower().strip()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda")
    if d == "mps":
        return torch.device("mps")
    return torch.device("cpu")


def check_environment_phase_b(cfg: TrainConfig) -> torch.device:
    resolved = resolve_device(cfg.device)
    ds = cfg.device.lower().strip()
    if ds == "cuda" and not torch.cuda.is_available():
        print("[fatal] device=cuda unavailable. Use cpu/mps/auto or install CUDA torch.\ndata_dir=", cfg.data_dir, file=sys.stderr)
        sys.exit(1)
    if ds == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        print("[fatal] device=mps unavailable.\ndata_dir=", cfg.data_dir, file=sys.stderr)
        sys.exit(1)
    lb = cfg.log_backend.lower().strip()
    if lb == "tensorboard":
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            print("[fatal] log_backend=tensorboard. pip install tensorboard", file=sys.stderr)
            sys.exit(1)
    elif lb == "wandb":
        try:
            import wandb  # noqa: F401
        except ImportError:
            print("[fatal] log_backend=wandb. pip install wandb", file=sys.stderr)
            sys.exit(1)
    tok = cfg.tokenizer.lower().strip()
    if tok in ("tiktoken", "bpe"):
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            print("[fatal] tokenizer=tiktoken. pip install tiktoken", file=sys.stderr)
            sys.exit(1)
    data_path = Path(cfg.data_dir).expanduser()
    if not data_path.exists():
        print("[fatal] data_dir missing:", data_path, "device=", cfg.device, file=sys.stderr)
        sys.exit(1)
    if data_path.is_dir() and not list(data_path.glob("*.txt")):
        print("[fatal] no .txt in:", data_path, file=sys.stderr)
        sys.exit(1)
    return resolved


def setup_device_and_threads(device: torch.device, num_threads: int) -> None:
    if num_threads and num_threads > 0:
        torch.set_num_threads(int(num_threads))
        try:
            torch.set_num_interop_threads(max(1, int(num_threads) // 2))
        except Exception:
            pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_rng_state(device: torch.device) -> dict[str, Any]:
    st: dict[str, Any] = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "random": random.getstate()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["random"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class CharTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class TiktokenTokenizer:
    def __init__(self, encoding_name: str) -> None:
        import tiktoken
        self._enc = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        return int(self._enc.n_vocab)

    def encode(self, s: str) -> List[int]:
        return self._enc.encode(s, allowed_special="all")

    def decode(self, ids: Sequence[int]) -> str:
        return self._enc.decode([int(i) for i in ids])


def load_text_corpus(data_dir: str) -> Tuple[str, dict[str, Any]]:
    p = Path(data_dir).expanduser()
    meta: dict[str, Any] = {"paths": [], "num_files": 0}
    if p.is_file():
        text = p.read_text(encoding="utf-8", errors="replace")
        meta["paths"] = [str(p)]
        meta["num_files"] = 1
        return text, meta
    parts: List[str] = []
    for fp in sorted(p.glob("*.txt")):
        parts.append(fp.read_text(encoding="utf-8", errors="replace"))
        meta["paths"].append(str(fp))
    meta["num_files"] = len(meta["paths"])
    return "".join(parts), meta


def build_tokenizer(kind: str, text: str, tiktoken_encoding: str) -> Union[CharTokenizer, TiktokenTokenizer]:
    k = kind.lower().strip()
    if k in ("char", "character"):
        return CharTokenizer(text)
    if k in ("tiktoken", "bpe"):
        return TiktokenTokenizer(tiktoken_encoding)
    raise ValueError(f"Unknown tokenizer: {kind}")


class BlockDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        if data.numel() <= block_size:
            raise ValueError(f"Corpus too short for block_size={block_size} tokens={data.numel()}")
        self.data = data
        self.block_size = int(block_size)
        usable = self.data.numel() - self.block_size
        self._starts = list(range(0, usable, self.block_size))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self._starts[idx]
        chunk = self.data[s : s + self.block_size + 1]
        x = chunk[:-1].to(dtype=torch.long)
        y = chunk[1:].to(dtype=torch.long)
        return x, y


def build_dataloaders(
    cfg: TrainConfig, device: torch.device, encoded: torch.Tensor
) -> Tuple[DataLoader, DataLoader, dict[str, Any]]:
    n = int(encoded.numel())
    n_train = int(n * float(cfg.train_ratio))
    n_val = int(n * float(cfg.val_ratio))
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    if n_train <= cfg.block_size or n_val <= cfg.block_size:
        raise ValueError(f"Bad split n={n} n_train={n_train} n_val={n_val} block={cfg.block_size}")
    train_data = encoded[:n_train].contiguous()
    val_data = encoded[n_train : n_train + n_val].contiguous()
    pin = bool(cfg.pin_memory) and device.type == "cuda"
    pw = bool(cfg.num_workers > 0)
    train_ds = BlockDataset(train_data, cfg.block_size)
    val_ds = BlockDataset(val_data, cfg.block_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=pw,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=pw,
        drop_last=False,
    )
    info = {
        "n_tokens_total": n,
        "n_train_tokens": int(train_data.numel()),
        "n_val_tokens": int(val_data.numel()),
        "train_blocks": len(train_ds),
        "val_batches_est": max(1, math.ceil(len(val_ds) / cfg.batch_size)),
    }
    return train_loader, val_loader, info


def batch_stream(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    it = iter(loader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(loader)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TrainConfig, n_embd: int, n_head: int) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd, bias=cfg.bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=cfg.bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: TrainConfig, n_embd: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=cfg.bias)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, cfg: TrainConfig, n_embd: int, n_head: int) -> None:
        super().__init__()
        self.ln1 = LayerNorm(n_embd, cfg.bias)
        self.attn = CausalSelfAttention(cfg, n_embd, n_head)
        self.ln2 = LayerNorm(n_embd, cfg.bias)
        self.mlp = MLP(cfg, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: TrainConfig, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(vocab_size)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(self.vocab_size, cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([Block(cfg, cfg.n_embd, cfg.n_head) for _ in range(cfg.n_layer)]),
                ln_f=LayerNorm(cfg.n_embd, cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, self.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.size()
        assert t <= self.cfg.block_size
        pos = torch.arange(0, t, device=idx.device, dtype=torch.long)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def learning_rate_for_step(cfg: TrainConfig, step: int) -> float:
    if step < cfg.warmup_iters:
        return cfg.lr * float(step + 1) / float(max(1, cfg.warmup_iters))
    if cfg.max_iters <= cfg.warmup_iters:
        return cfg.min_lr
    decay_steps = max(1, cfg.max_iters - cfg.warmup_iters)
    t = float(step - cfg.warmup_iters) / float(decay_steps)
    t = min(max(t, 0.0), 1.0)
    return cfg.min_lr + (0.5 * (1.0 + math.cos(math.pi * t))) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def evaluate(model: GPT, val_loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    losses: List[float] = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def compute_grad_norm_l2(model: nn.Module) -> float:
    """Total L2 norm of all parameter gradients (after backward, before step)."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += float(p.grad.data.norm(2).item() ** 2)
    return total_norm_sq**0.5


def train_step(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: TrainConfig,
) -> Tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        with torch.cuda.amp.autocast(device_type="cuda", dtype=torch.float16):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = compute_grad_norm_l2(model)
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        _, loss = model(x, y)
        loss.backward()
        grad_norm = compute_grad_norm_l2(model)
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    return float(loss.item()), float(grad_norm)


@torch.no_grad()
def generate_sample_text(
    model: GPT,
    tokenizer: Union[CharTokenizer, TiktokenTokenizer],
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> str:
    model.eval()
    start_ids = tokenizer.encode(prompt or "\n")
    idx = torch.tensor([start_ids], dtype=torch.long, device=device)
    for _ in range(int(max_new_tokens)):
        idx_cond = idx[:, -model.cfg.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
    return tokenizer.decode(idx[0].tolist())


def save_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    step: int,
    best_val_loss: float,
    cfg: TrainConfig,
    rng_state: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "step": int(step),
        "best_val_loss": float(best_val_loss),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": asdict(cfg),
        "rng": rng_state,
        "extra": extra or {},
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
) -> dict[str, Any]:
    # Full checkpoints include numpy RNG etc.; PyTorch 2.6+ defaults weights_only=True.
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if payload.get("rng"):
        restore_rng_state(payload["rng"], device)
    return payload


class LoggerBackend:
    def __init__(self, backend: str, out_dir: Path, run_name: str) -> None:
        self.backend = backend.lower().strip()
        self._tb = None
        self._wandb = None
        if self.backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(log_dir=str(out_dir / "tb"))
        elif self.backend == "wandb":
            import wandb
            self._wandb = wandb.init(project="train_gpt", name=run_name, dir=str(out_dir))

    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        if self.backend == "none":
            return
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, step)
        if self._wandb is not None:
            import wandb
            wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            import wandb
            wandb.finish()


def _config_groups() -> dict[str, list[str]]:
    return {
        "infrastructure": [
            "device",
            "num_threads",
            "seed",
            "tokenizer",
            "tiktoken_encoding",
            "log_backend",
            "num_workers",
            "pin_memory",
            "amp",
        ],
        "training": [
            "batch_size",
            "lr",
            "max_iters",
            "weight_decay",
            "warmup_iters",
            "min_lr",
            "grad_clip",
            "adam_beta1",
            "adam_beta2",
            "eval_interval",
            "checkpoint_interval",
            "early_stop_patience",
            "resume",
            "checkpoint",
        ],
        "model": ["n_layer", "n_head", "n_embd", "block_size", "dropout", "bias"],
        "io": [
            "data_dir",
            "out_dir",
            "train_ratio",
            "val_ratio",
            "sample_interval",
            "sample_prompt",
            "sample_max_new_tokens",
            "append_samples_to_file",
        ],
    }


def print_startup_report(
    cfg: TrainConfig,
    runtime: dict[str, Any],
    *,
    out_dir: Path,
    device: torch.device,
    resume_info: Optional[dict[str, Any]] = None,
    startup_log_path: Optional[Path] = None,
) -> None:
    lines: List[str] = []
    flat = asdict(cfg)
    groups = _config_groups()
    grouped_keys = set().union(*groups.values())
    stray = sorted(set(flat.keys()) - grouped_keys)
    lines.append("=== Resolved configuration (TrainConfig) ===")
    for gname, keys in groups.items():
        lines.append(f"-- {gname} --")
        for k in keys:
            if k in flat:
                lines.append(f"  {k}: {flat[k]!r}")
    if stray:
        lines.append("-- other --")
        for k in stray:
            lines.append(f"  {k}: {flat[k]!r}")
    lines.append("=== Runtime ===")
    lines.append(f"  script_version: {__version__!r}")
    lines.append(f"  script_author: {__author__!r}")
    lines.append(f"  python_version: {sys.version.split()[0]!r}")
    lines.append(f"  sys_version_one_line: {sys.version.strip()!r}")
    lines.append(f"  torch_version: {runtime.get('torch_version', 'n/a')!r}")
    lines.append(f"  resolved_device: {str(device)!r}")
    lines.append(f"  torch_num_threads: {torch.get_num_threads()}")
    if device.type == "cuda" and torch.cuda.is_available():
        lines.append(f"  cuda_device_name: {torch.cuda.get_device_name(device)}")
        di = device.index if device.index is not None else 0
        lines.append(f"  cuda_device_index: {di}")
    elif device.type == "mps":
        lines.append("  mps: true")
    else:
        lines.append("  cuda_device_name: n/a")
    lines.append(f"  data_dir_resolved: {str(Path(cfg.data_dir).expanduser())!r}")
    lines.append(f"  corpus_chars: {runtime.get('corpus_num_chars', 'n/a')}")
    lines.append(f"  corpus_utf8_bytes: {runtime.get('corpus_utf8_bytes', 'n/a')}")
    lines.append(f"  vocab_size: {runtime.get('vocab_size', 'n/a')}")
    lines.append(f"  tokenizer_impl: {runtime.get('tokenizer_impl', 'n/a')!r}")
    lines.append(f"  train_blocks: {runtime.get('train_blocks', 'n/a')}")
    lines.append(f"  n_train_tokens: {runtime.get('n_train_tokens', 'n/a')}")
    lines.append(f"  n_val_tokens: {runtime.get('n_val_tokens', 'n/a')}")
    lines.append(f"  pin_memory_effective: {runtime.get('pin_memory_effective', 'n/a')}")
    lines.append(f"  model_n_layer: {cfg.n_layer}")
    lines.append(f"  model_n_head: {cfg.n_head}")
    lines.append(f"  model_n_embd: {cfg.n_embd}")
    lines.append(f"  model_block_size: {cfg.block_size}")
    lines.append(f"  num_parameters_total: {runtime.get('num_parameters_total', 'n/a')}")
    lines.append(f"  num_parameters_trainable: {runtime.get('num_parameters_trainable', 'n/a')}")
    lines.append(f"  out_dir: {str(out_dir)!r}")
    lines.append(f"  checkpoint_latest_path: {str(out_dir / 'latest.pt')!r}")
    lines.append(f"  checkpoint_best_path: {str(out_dir / 'best.pt')!r}")
    if resume_info:
        lines.append("=== Resume ===")
        for k in sorted(resume_info.keys()):
            lines.append(f"  {k}: {resume_info[k]!r}")
    msg = "\n".join(lines) + "\n"
    print(msg, end="")
    if startup_log_path is not None:
        startup_log_path.parent.mkdir(parents=True, exist_ok=True)
        startup_log_path.write_text(msg, encoding="utf-8")


def apply_cli_to_config(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    d = asdict(cfg)
    for ff in fields(TrainConfig):
        name = ff.name
        if name in ("resume", "amp"):
            continue
        v = getattr(args, name, None)
        if v is None:
            continue
        d[name] = v
    if args.resume:
        d["resume"] = True
    if args.amp:
        d["amp"] = True
    return TrainConfig(**d)


def dump_resolved_config_yaml(cfg: TrainConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=True, allow_unicode=True)


def main(args: argparse.Namespace) -> None:
    phase_a = check_environment_phase_a()
    cfg = TrainConfig()
    cfg = merge_yaml_into_config(cfg, args.config or None)
    cfg = apply_cli_to_config(cfg, args)

    if cfg.eval_interval < 1 or cfg.checkpoint_interval < 1:
        print("[fatal] eval_interval and checkpoint_interval must be >= 1", file=sys.stderr)
        sys.exit(1)
    device = check_environment_phase_b(cfg)
    if cfg.n_embd % cfg.n_head != 0:
        print(f"[fatal] n_embd % n_head must be 0: {cfg.n_embd} {cfg.n_head}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(cfg.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_resolved_config_yaml(cfg, out_dir / "config_resolved.yaml")

    setup_device_and_threads(device, cfg.num_threads)
    set_seed(cfg.seed)
    if cfg.amp and device.type != "cuda":
        print("[warn] AMP needs CUDA; disabling AMP.", file=sys.stderr)
        cfg = dataclasses.replace(cfg, amp=False)

    text, corpus_meta = load_text_corpus(cfg.data_dir)
    tokenizer = build_tokenizer(cfg.tokenizer, text, cfg.tiktoken_encoding)
    encoded = torch.tensor([int(x) for x in tokenizer.encode(text)], dtype=torch.long)
    tok_name = "char" if isinstance(tokenizer, CharTokenizer) else f"tiktoken({cfg.tiktoken_encoding})"

    train_loader, val_loader, split_info = build_dataloaders(cfg, device, encoded)
    pin_effective = bool(cfg.pin_memory) and device.type == "cuda"
    model = GPT(cfg, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if device.type == "cuda" and cfg.amp:
        scaler = torch.cuda.amp.GradScaler()

    step = 0
    best_val_loss = float("inf")
    resume_info: Optional[dict[str, Any]] = None
    ckpt_path = Path(cfg.checkpoint).expanduser() if cfg.checkpoint else out_dir / "latest.pt"
    if cfg.resume:
        if not ckpt_path.is_file():
            print(f"[fatal] --resume but checkpoint missing: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        payload = load_checkpoint(
            ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, device=device
        )
        step = int(payload.get("step", 0))
        best_val_loss = float(payload.get("best_val_loss", float("inf")))
        resume_info = {"loaded_checkpoint": str(ckpt_path), "restored_step": step, "restored_best_val_loss": best_val_loss}

    logger = LoggerBackend(cfg.log_backend, out_dir, run_name=out_dir.name)
    runtime: dict[str, Any] = {
        **phase_a,
        "corpus_num_chars": len(text),
        "corpus_utf8_bytes": len(text.encode("utf-8")),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_impl": tok_name,
        **split_info,
        "pin_memory_effective": pin_effective,
        "num_parameters_total": count_parameters(model, False),
        "num_parameters_trainable": count_parameters(model, True),
        "corpus_meta": corpus_meta,
    }

    print_startup_report(
        cfg,
        runtime,
        out_dir=out_dir,
        device=device,
        resume_info=resume_info,
        startup_log_path=out_dir / "startup_log.txt",
    )

    model.train()
    train_iter = batch_stream(train_loader)
    bad_epochs = 0
    t0 = time.time()
    eval_records: List[dict[str, Any]] = []

    try:
        while step < cfg.max_iters:
            lr_now = learning_rate_for_step(cfg, step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            x, y = next(train_iter)
            rng_state = collect_rng_state(device)
            x = x.to(device, non_blocking=pin_effective)
            y = y.to(device, non_blocking=pin_effective)
            loss_tr, grad_norm = train_step(model, x, y, optimizer, scaler, cfg)

            if step % cfg.eval_interval == 0 or step == 0:
                val_loss = evaluate(model, val_loader, device)
                ppl = math.exp(val_loss) if val_loss == val_loss else float("nan")
                elapsed = time.time() - t0
                print(
                    f"[step {step}/{cfg.max_iters}] train_loss={loss_tr:.4f} grad_norm={grad_norm:.4f} "
                    f"val_loss={val_loss:.4f} ppl={ppl:.2f} lr={lr_now:.2e} ({elapsed:.1f}s)"
                )
                eval_records.append(
                    {
                        "step": step,
                        "train_loss": loss_tr,
                        "grad_norm": grad_norm,
                        "val_loss": val_loss,
                        "ppl": ppl,
                        "lr": lr_now,
                        "elapsed_s": elapsed,
                    }
                )
                logger.log_scalars(
                    step,
                    {
                        "train/loss": loss_tr,
                        "train/grad_norm": grad_norm,
                        "val/loss": val_loss,
                        "val/perplexity": ppl,
                        "train/lr": lr_now,
                    },
                )
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    bad_epochs = 0
                    save_checkpoint(
                        out_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=step,
                        best_val_loss=best_val_loss,
                        cfg=cfg,
                        rng_state=rng_state,
                        extra={"note": "best_val"},
                    )
                else:
                    bad_epochs += 1
                    if cfg.early_stop_patience and bad_epochs >= cfg.early_stop_patience:
                        print(f"[early_stop] no val improvement for {bad_epochs} evals; best={best_val_loss:.4f}")
                        break

            if step > 0 and step % cfg.checkpoint_interval == 0:
                save_checkpoint(
                    out_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    best_val_loss=best_val_loss,
                    cfg=cfg,
                    rng_state=rng_state,
                    extra={"note": "latest"},
                )

            if cfg.sample_interval and (step % cfg.sample_interval == 0):
                gen = generate_sample_text(
                    model, tokenizer, device, prompt=cfg.sample_prompt or "", max_new_tokens=cfg.sample_max_new_tokens
                )
                banner = f"\n=== sample @ step {step} ===\n"
                print(banner + gen + "\n")
                if cfg.append_samples_to_file:
                    with (out_dir / "samples.txt").open("a", encoding="utf-8") as sf:
                        sf.write(banner)
                        sf.write(gen + "\n")

            step += 1
    finally:
        save_checkpoint(
            out_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            best_val_loss=best_val_loss,
            cfg=cfg,
            rng_state=collect_rng_state(device),
            extra={"note": "final"},
        )
        logger.close()

    print("\n=== Eval history (every eval_interval) ===")
    for r in eval_records:
        print(
            f"  step={r['step']:<6} train_loss={r['train_loss']:.4f} grad_norm={r['grad_norm']:.4f} "
            f"val_loss={r['val_loss']:.4f} ppl={r['ppl']:.2f} lr={r['lr']:.2e} elapsed_s={r['elapsed_s']:.1f}s"
        )
    print(f"\n[done] last_step={step} best_val_loss={best_val_loss:.4f} eval_points={len(eval_records)}")


if __name__ == "__main__":
    _args = parse_cli()
    main(_args)
