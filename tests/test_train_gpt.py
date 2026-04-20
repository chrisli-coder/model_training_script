from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt as tg

requires_cuda = pytest.mark.skipif(not tg.torch.cuda.is_available(), reason="CUDA not available")
requires_cuda_bf16 = pytest.mark.skipif(
    not tg.torch.cuda.is_available() or not tg.torch.cuda.is_bf16_supported(),
    reason="CUDA bfloat16 not available",
)


@pytest.fixture()
def test_assets(tmp_path: Path) -> dict[str, Path]:
    text = (
        "hello world\n"
        "hello there world\n"
        "tiny gpt trainer test corpus\n"
        "memmap data path validation\n"
    ) * 20
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")

    chars = sorted(set(text))
    char_vocab_txt = tmp_path / "char_vocab.txt"
    char_vocab_txt.write_text("".join(chars), encoding="utf-8")
    char_vocab_single_line_newline = tmp_path / "char_vocab_single_line_newline.txt"
    char_vocab_single_line_newline.write_text("abc\n", encoding="utf-8")
    line_chars = [ch for ch in chars if ch != "\n"]
    char_vocab_lines = tmp_path / "char_vocab_lines.txt"
    char_vocab_lines.write_text("\n".join(line_chars), encoding="utf-8")

    char_vocab_json = tmp_path / "char_vocab.json"
    char_vocab_json.write_text(json.dumps({"chars": chars}, ensure_ascii=True), encoding="utf-8")

    invalid_vocab_json = tmp_path / "invalid_vocab.json"
    invalid_vocab_json.write_text(json.dumps({"not_chars": chars}, ensure_ascii=True), encoding="utf-8")

    vocab = {
        "[UNK]": 0,
        "hello": 1,
        "world": 2,
        "there": 3,
        "tiny": 4,
        "gpt": 5,
        "trainer": 6,
        "test": 7,
        "corpus": 8,
        "memmap": 9,
        "data": 10,
        "path": 11,
        "validation": 12,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer_json = tmp_path / "tokenizer.json"
    tok.save(str(tokenizer_json))

    train_ids = np.tile(np.array([1, 2, 1, 3, 2, 4, 5, 6, 7, 8], dtype=np.uint16), 64)
    val_ids = np.tile(np.array([9, 10, 11, 12, 1, 2], dtype=np.uint16), 32)
    train_bin = tmp_path / "train.bin"
    val_bin = tmp_path / "val.bin"
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)

    return {
        "corpus": corpus,
        "char_vocab_txt": char_vocab_txt,
        "char_vocab_single_line_newline": char_vocab_single_line_newline,
        "char_vocab_lines": char_vocab_lines,
        "char_vocab_json": char_vocab_json,
        "invalid_vocab_json": invalid_vocab_json,
        "tokenizer_json": tokenizer_json,
        "train_bin": train_bin,
        "val_bin": val_bin,
        "data_dir": tmp_path,
    }


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, str(ROOT / "train_gpt.py"), *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def tiny_cfg(**kwargs: object) -> tg.TrainConfig:
    cfg = tg.TrainConfig(
        device="cpu",
        batch_size=2,
        accumulation_steps=1,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=8,
        max_iters=2,
        eval_interval=1,
        checkpoint_interval=1,
        sample_interval=0,
        num_workers=0,
        out_dir="runs/test",
    )
    for key, value in kwargs.items():
        setattr(cfg, key, value)
    return cfg


def test_load_char_vocab_rejects_invalid_json(test_assets: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="top-level 'chars' list"):
        tg.load_char_vocab(str(test_assets["invalid_vocab_json"]))


def test_load_char_vocab_accepts_json_and_text(test_assets: dict[str, Path]) -> None:
    tok_json = tg.load_char_vocab(str(test_assets["char_vocab_json"]))
    tok_txt = tg.load_char_vocab(str(test_assets["char_vocab_txt"]))
    assert tok_json.vocab_size == tok_txt.vocab_size
    assert tok_json.encode("hello") == tok_txt.encode("hello")


def test_load_char_vocab_accepts_one_char_per_line(test_assets: dict[str, Path]) -> None:
    tok_lines = tg.load_char_vocab(str(test_assets["char_vocab_lines"]))
    assert "\n" not in tok_lines.stoi
    assert tok_lines.encode("hello world") == [tok_lines.stoi[c] for c in "hello world"]


def test_load_char_vocab_single_line_with_trailing_newline(test_assets: dict[str, Path]) -> None:
    tok = tg.load_char_vocab(str(test_assets["char_vocab_single_line_newline"]))
    assert "\n" not in tok.stoi
    assert tok.encode("abc") == [tok.stoi["a"], tok.stoi["b"], tok.stoi["c"]]


def test_token_window_dataset_val_matches_sequential_windows() -> None:
    data = np.arange(20, dtype=np.int64)
    block = 4
    ds = tg.TokenWindowDataset(data, block, randomize=False)
    assert len(ds) == (len(data) - block - 1) + 1
    for idx in range(len(ds)):
        x, y = ds[idx]
        exp_x = tg.torch.tensor(data[idx : idx + block], dtype=tg.torch.long)
        exp_y = tg.torch.tensor(data[idx + 1 : idx + block + 1], dtype=tg.torch.long)
        assert tg.torch.equal(x, exp_x)
        assert tg.torch.equal(y, exp_y)


def test_token_window_dataset_val_dataloader_order_matches_indices(test_assets: dict[str, Path]) -> None:
    cfg = tiny_cfg(
        data_dir=str(test_assets["data_dir"]),
        data_format="bin",
        tokenizer=str(test_assets["tokenizer_json"]),
        token_dtype="uint16",
    )
    tokenizer = tg.build_tokenizer(str(test_assets["tokenizer_json"]), text=None, tiktoken_encoding="gpt2")
    train_ds, val_ds, _, _ = tg.prepare_datasets(cfg, tokenizer)
    _, val_loader, _ = tg.build_dataloaders(cfg, tg.torch.device("cpu"), train_ds, val_ds)
    idx = 0
    for xb, yb in val_loader:
        bs = xb.shape[0]
        for j in range(bs):
            xr, yr = val_ds[idx]
            assert tg.torch.equal(xb[j], xr)
            assert tg.torch.equal(yb[j], yr)
            idx += 1
        break
    assert idx > 0


def test_token_window_dataset_train_len_and_shapes() -> None:
    data = np.zeros(100, dtype=np.int64)
    block = 8
    ds = tg.TokenWindowDataset(data, block, randomize=True)
    assert len(ds) == tg.RANDOM_WINDOW_DATASET_LEN
    assert ds.num_windows == 100 - block
    x, y = ds[0]
    assert tuple(x.shape) == (block,)
    assert tuple(y.shape) == (block,)


def test_token_window_dataset_rejects_short_corpus() -> None:
    data = np.arange(5, dtype=np.int64)
    with pytest.raises(ValueError, match="Corpus too short"):
        tg.TokenWindowDataset(data, block_size=8, randomize=True)


def test_seed_dataloader_worker_produces_distinct_rng_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    def fake_seed(seed: int) -> None:
        seen.append(seed)

    monkeypatch.setattr(tg.random, "seed", fake_seed)
    monkeypatch.setattr(tg.np.random, "seed", fake_seed)
    monkeypatch.setattr(tg.torch, "initial_seed", lambda: 1234)

    tg.seed_dataloader_worker(0)
    tg.seed_dataloader_worker(1)

    assert seen == [1234, 1234, 1235, 1235]


def test_prepare_text_datasets(test_assets: dict[str, Path]) -> None:
    cfg = tiny_cfg(data_dir=str(test_assets["corpus"]), data_format="text", tokenizer="char")
    tokenizer = tg.build_tokenizer("char", text=test_assets["corpus"].read_text(encoding="utf-8"), tiktoken_encoding="gpt2")
    train_ds, val_ds, info, corpus_info = tg.prepare_datasets(cfg, tokenizer)
    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert info["n_train_tokens"] > info["n_val_tokens"] > 0
    assert corpus_info["corpus_num_chars"] > 0


def test_prepare_bin_datasets(test_assets: dict[str, Path]) -> None:
    cfg = tiny_cfg(
        data_dir=str(test_assets["data_dir"]),
        data_format="bin",
        tokenizer=str(test_assets["tokenizer_json"]),
        token_dtype="uint16",
    )
    tokenizer = tg.build_tokenizer(str(test_assets["tokenizer_json"]), text=None, tiktoken_encoding="gpt2")
    train_ds, val_ds, info, corpus_info = tg.prepare_datasets(cfg, tokenizer)
    x, y = train_ds[0]
    assert x.dtype == tg.torch.long
    assert y.dtype == tg.torch.long
    assert tuple(x.shape) == (cfg.block_size,)
    assert info["n_tokens_total"] == info["n_train_tokens"] + info["n_val_tokens"]
    assert corpus_info["corpus_meta"]["format"] == "bin"


def test_train_step_with_accumulation_updates_model(test_assets: dict[str, Path]) -> None:
    cfg = tiny_cfg(
        data_dir=str(test_assets["data_dir"]),
        data_format="bin",
        tokenizer=str(test_assets["tokenizer_json"]),
        accumulation_steps=2,
    )
    tokenizer = tg.build_tokenizer(str(test_assets["tokenizer_json"]), text=None, tiktoken_encoding="gpt2")
    train_ds, val_ds, _, _ = tg.prepare_datasets(cfg, tokenizer)
    train_loader, _, _ = tg.build_dataloaders(cfg, tg.torch.device("cpu"), train_ds, val_ds)
    train_iter = tg.batch_stream(train_loader)
    model = tg.GPT(cfg, tokenizer.vocab_size)
    optimizer, _, _ = tg.build_optimizer(model, cfg, tg.torch.device("cpu"))
    before = [p.detach().clone() for p in model.parameters()]
    loss, grad_norm = tg.train_step(model, train_iter, tg.torch.device("cpu"), False, optimizer, None, cfg)
    after = list(model.parameters())
    assert loss > 0
    assert grad_norm > 0
    assert any(not tg.torch.equal(b, a.detach()) for b, a in zip(before, after))


def test_cli_text_char_smoke(test_assets: dict[str, Path], tmp_path: Path) -> None:
    out_dir = tmp_path / "text_char"
    proc = run_cli(
        tmp_path,
        "--device", "cpu",
        "--data_dir", str(test_assets["corpus"]),
        "--data_format", "text",
        "--tokenizer", "char",
        "--n_layer", "2",
        "--n_head", "2",
        "--n_embd", "32",
        "--block_size", "8",
        "--batch_size", "2",
        "--max_iters", "2",
        "--eval_interval", "1",
        "--checkpoint_interval", "1",
        "--sample_interval", "0",
        "--out_dir", str(out_dir),
    )
    assert proc.returncode == 0, proc.stdout
    assert "[done]" in proc.stdout
    assert (out_dir / "latest.pt").is_file()


def test_cli_bin_json_and_resume_smoke(test_assets: dict[str, Path], tmp_path: Path) -> None:
    out_dir = tmp_path / "bin_json"
    common = [
        "--device", "cpu",
        "--data_dir", str(test_assets["data_dir"]),
        "--data_format", "bin",
        "--tokenizer", str(test_assets["tokenizer_json"]),
        "--n_layer", "2",
        "--n_head", "2",
        "--n_embd", "32",
        "--block_size", "8",
        "--batch_size", "2",
        "--max_iters", "4",
        "--eval_interval", "1",
        "--checkpoint_interval", "1",
        "--sample_interval", "0",
        "--accumulation_steps", "2",
        "--out_dir", str(out_dir),
    ]
    first = run_cli(tmp_path, *common)
    assert first.returncode == 0, first.stdout
    assert "accumulation_steps: 2" in first.stdout
    assert "last_step=4" in first.stdout
    payload = tg.torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 4
    second = run_cli(tmp_path, *(common + ["--resume"]))
    assert second.returncode == 0, second.stdout
    assert "loaded_checkpoint" in second.stdout


def test_main_saves_post_step_rng_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tg.TrainConfig(
        device="cpu",
        batch_size=2,
        accumulation_steps=2,
        max_iters=2,
        eval_interval=10,
        checkpoint_interval=10,
        sample_interval=0,
        out_dir=str(tmp_path / "run"),
        data_dir=str(tmp_path),
        data_format="bin",
        tokenizer="char",
        vocab_file=str(tmp_path / "char_vocab.txt"),
    )
    args = tg.argparse.Namespace(config="", **tg.asdict(cfg))
    args.resume = False
    args.amp = False
    args.gradient_checkpointing = False

    saved: list[tuple[str, int, str]] = []
    collect_calls = iter([{"marker": "after_step_1"}, {"marker": "after_step_2"}, {"marker": "final"}])

    class DummyModel:
        def train(self) -> None:
            return None

        def parameters(self):
            return []

        def to(self, device):
            return self

    class DummyOptimizer:
        param_groups = [{"lr": 0.0}]

        def state_dict(self):
            return {}

    class DummyLogger:
        def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(tg, "check_environment_phase_a", lambda: {"torch_version": "test"})
    monkeypatch.setattr(tg, "merge_yaml_into_config", lambda cfg0, yaml_path: cfg0)
    monkeypatch.setattr(tg, "apply_cli_to_config", lambda cfg0, args0: cfg)
    monkeypatch.setattr(tg, "check_environment_phase_b", lambda cfg0: tg.torch.device("cpu"))
    monkeypatch.setattr(tg, "dump_resolved_config_yaml", lambda cfg0, path: None)
    monkeypatch.setattr(tg, "setup_device_and_threads", lambda device, num_threads: None)
    monkeypatch.setattr(tg, "set_seed", lambda seed: None)
    monkeypatch.setattr(tg, "build_tokenizer", lambda *a, **k: type("Tok", (), {"vocab_size": 8})())
    monkeypatch.setattr(tg, "tokenizer_runtime_name", lambda cfg0, tokenizer: "char")
    monkeypatch.setattr(tg, "prepare_datasets", lambda cfg0, tokenizer: (object(), object(), {"n_train_tokens": 10, "n_val_tokens": 2, "train_blocks": 2}, {"corpus_num_chars": 0, "corpus_utf8_bytes": 0}))
    monkeypatch.setattr(tg, "build_dataloaders", lambda cfg0, device, train_ds, val_ds: (object(), object(), {"val_batches_est": 1}))
    monkeypatch.setattr(tg, "GPT", lambda cfg0, vocab_size: DummyModel())
    monkeypatch.setattr(tg, "build_optimizer", lambda model, cfg0, device: (DummyOptimizer(), "adamw", None))
    monkeypatch.setattr(tg, "LoggerBackend", lambda backend, out_dir, run_name: DummyLogger())
    monkeypatch.setattr(tg, "print_startup_report", lambda *a, **k: None)
    monkeypatch.setattr(tg, "batch_stream", lambda loader: iter(()))
    monkeypatch.setattr(tg, "train_step", lambda *a, **k: (1.0, 0.5))
    monkeypatch.setattr(tg, "evaluate", lambda *a, **k: 1.0)
    monkeypatch.setattr(tg, "collect_rng_state", lambda device: next(collect_calls))
    monkeypatch.setattr(
        tg,
        "save_checkpoint",
        lambda path, **kwargs: saved.append((Path(path).name, kwargs["step"], kwargs["rng_state"]["marker"])),
    )

    tg.main(args)

    assert ("best.pt", 1, "after_step_1") in saved
    assert ("latest.pt", 2, "final") == saved[-1]


def test_accumulation_does_not_reduce_optimizer_update_budget(test_assets: dict[str, Path], tmp_path: Path) -> None:
    out_dir = tmp_path / "accum_budget"
    proc = run_cli(
        tmp_path,
        "--device", "cpu",
        "--data_dir", str(test_assets["data_dir"]),
        "--data_format", "bin",
        "--tokenizer", str(test_assets["tokenizer_json"]),
        "--n_layer", "2",
        "--n_head", "2",
        "--n_embd", "32",
        "--block_size", "8",
        "--batch_size", "2",
        "--max_iters", "5",
        "--eval_interval", "2",
        "--checkpoint_interval", "2",
        "--sample_interval", "0",
        "--accumulation_steps", "4",
        "--out_dir", str(out_dir),
    )
    assert proc.returncode == 0, proc.stdout
    assert "last_step=5" in proc.stdout
    payload = tg.torch.load(out_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == 5


# --- Unit: RoPE helpers ---


def test_unit_rotate_half_swaps_halves_and_negates() -> None:
    x = tg.torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = tg.rotate_half(x)
    tg.torch.testing.assert_close(y, tg.torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_unit_apply_rotary_pos_emb_identity_when_sin_is_zero() -> None:
    B, H, T, D = 1, 2, 4, 8
    q = tg.torch.randn(B, H, T, D)
    k = tg.torch.randn(B, H, T, D)
    cos = tg.torch.ones(1, 1, T, D)
    sin = tg.torch.zeros(1, 1, T, D)
    q2, k2 = tg.apply_rotary_pos_emb(q, k, cos, sin)
    tg.torch.testing.assert_close(q2, q)
    tg.torch.testing.assert_close(k2, k)


def test_unit_rotary_embedding_output_shape_and_dtype() -> None:
    rope = tg.RotaryEmbedding(dim=8, max_position_embeddings=32, base=10000.0)
    cos, sin = rope.forward(7, dtype=tg.torch.bfloat16, device=tg.torch.device("cpu"))
    assert cos.shape == (1, 1, 7, 8)
    assert sin.shape == (1, 1, 7, 8)
    assert cos.dtype == tg.torch.bfloat16
    assert sin.dtype == tg.torch.bfloat16


def test_unit_cuda_autocast_dtype_for_config() -> None:
    base = tiny_cfg()
    assert tg.cuda_autocast_dtype_for_config(base) == tg.torch.float16
    assert tg.cuda_autocast_dtype_for_config(dataclasses.replace(base, amp_dtype="  BFLOAT16  ")) == tg.torch.bfloat16
    assert tg.cuda_autocast_dtype_for_config(dataclasses.replace(base, amp_dtype="float16")) == tg.torch.float16


def test_unit_gpt_state_dict_has_rotary_not_wpe() -> None:
    cfg = tiny_cfg()
    m = tg.GPT(cfg, vocab_size=50)
    keys = set(m.state_dict().keys())
    assert "transformer.wpe.weight" not in keys
    assert hasattr(m, "rotary_emb")
    # RoPE cos/sin caches use register_buffer(..., persistent=False) and are omitted from state_dict.
    assert not any(k.startswith("rotary_emb.") for k in keys)


def test_unit_gpt_rejects_odd_head_dim() -> None:
    cfg = tiny_cfg(n_embd=30, n_head=2)
    with pytest.raises(AssertionError, match="even head_dim"):
        tg.GPT(cfg, vocab_size=50)


# --- Integration: model + RoPE + backward ---


def test_integration_gpt_forward_backward_with_gradient_checkpointing() -> None:
    cfg = tiny_cfg(gradient_checkpointing=True, dropout=0.0)
    m = tg.GPT(cfg, vocab_size=20)
    m.train()
    idx = tg.torch.randint(0, 20, (2, 6))
    targets = tg.torch.randint(0, 20, (2, 6))
    logits, loss = m(idx, targets)
    assert logits.shape == (2, 6, 20)
    assert loss is not None
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_integration_gpt_rope_base_changes_rotary_buffers() -> None:
    cfg_a = tiny_cfg(rope_base=10000.0)
    cfg_b = tiny_cfg(rope_base=5000.0)
    a = tg.GPT(cfg_a, 10)
    b = tg.GPT(cfg_b, 10)
    assert not tg.torch.allclose(a.rotary_emb.cos_cached, b.rotary_emb.cos_cached)


# --- Regression: checkpoint / layout contracts ---


def test_regression_legacy_wpe_in_checkpoint_is_ignored_strict_false() -> None:
    cfg = tiny_cfg()
    m = tg.GPT(cfg, vocab_size=10)
    state = m.state_dict()
    polluted = {**state, "transformer.wpe.weight": tg.torch.zeros(cfg.block_size, cfg.n_embd)}
    fresh = tg.GPT(cfg, vocab_size=10)
    inc = fresh.load_state_dict(polluted, strict=False)
    assert "transformer.wpe.weight" in inc.unexpected_keys
    for k, v in state.items():
        tg.torch.testing.assert_close(fresh.state_dict()[k], v)


# --- Smoke / E2E: CLI flags for RoPE + checkpointing ---


def test_smoke_cli_gradient_checkpointing_and_rope_base(test_assets: dict[str, Path], tmp_path: Path) -> None:
    out_dir = tmp_path / "rope_cli"
    proc = run_cli(
        tmp_path,
        "--device", "cpu",
        "--data_dir", str(test_assets["data_dir"]),
        "--data_format", "bin",
        "--tokenizer", str(test_assets["tokenizer_json"]),
        "--n_layer", "2",
        "--n_head", "2",
        "--n_embd", "32",
        "--block_size", "8",
        "--batch_size", "2",
        "--max_iters", "2",
        "--eval_interval", "1",
        "--checkpoint_interval", "1",
        "--sample_interval", "0",
        "--gradient_checkpointing",
        "--rope_base", "5000",
        "--out_dir", str(out_dir),
    )
    assert proc.returncode == 0, proc.stdout
    assert "rope_base" in proc.stdout
    assert "[done]" in proc.stdout


def test_e2e_generate_sample_text_roundtrip(test_assets: dict[str, Path]) -> None:
    cfg = tiny_cfg(
        data_dir=str(test_assets["data_dir"]),
        data_format="bin",
        tokenizer=str(test_assets["tokenizer_json"]),
    )
    tok = tg.build_tokenizer(str(test_assets["tokenizer_json"]), text=None, tiktoken_encoding="gpt2")
    m = tg.GPT(cfg, tok.vocab_size)
    m.eval()
    out = tg.generate_sample_text(m, tok, tg.torch.device("cpu"), prompt="hello", max_new_tokens=3)
    assert isinstance(out, str)


# --- System-ish: CUDA AMP paths (hardware-dependent) ---


@requires_cuda
def test_system_cuda_train_step_amp_fp16_with_scaler(test_assets: dict[str, Path]) -> None:
    cfg = tiny_cfg(
        device="cuda",
        amp=True,
        amp_dtype="float16",
        data_dir=str(test_assets["data_dir"]),
        data_format="bin",
        tokenizer=str(test_assets["tokenizer_json"]),
    )
    tok = tg.build_tokenizer(str(test_assets["tokenizer_json"]), text=None, tiktoken_encoding="gpt2")
    train_ds, val_ds, _, _ = tg.prepare_datasets(cfg, tok)
    train_loader, _, _ = tg.build_dataloaders(cfg, tg.torch.device("cuda"), train_ds, val_ds)
    it = tg.batch_stream(train_loader)
    model = tg.GPT(cfg, tok.vocab_size).to("cuda")
    opt, _, _ = tg.build_optimizer(model, cfg, tg.torch.device("cuda"))
    scaler = tg.torch.amp.GradScaler("cuda")
    loss, gn = tg.train_step(model, it, tg.torch.device("cuda"), True, opt, scaler, cfg)
    assert loss == loss and loss > 0
    assert gn == gn and gn >= 0


@requires_cuda_bf16
def test_system_cuda_train_step_amp_bfloat16_without_scaler(test_assets: dict[str, Path]) -> None:
    cfg = tiny_cfg(
        device="cuda",
        amp=True,
        amp_dtype="bfloat16",
        data_dir=str(test_assets["data_dir"]),
        data_format="bin",
        tokenizer=str(test_assets["tokenizer_json"]),
    )
    tok = tg.build_tokenizer(str(test_assets["tokenizer_json"]), text=None, tiktoken_encoding="gpt2")
    train_ds, val_ds, _, _ = tg.prepare_datasets(cfg, tok)
    train_loader, _, _ = tg.build_dataloaders(cfg, tg.torch.device("cuda"), train_ds, val_ds)
    it = tg.batch_stream(train_loader)
    model = tg.GPT(cfg, tok.vocab_size).to("cuda")
    opt, _, _ = tg.build_optimizer(model, cfg, tg.torch.device("cuda"))
    loss, gn = tg.train_step(model, it, tg.torch.device("cuda"), True, opt, None, cfg)
    assert loss == loss and loss > 0
    assert gn == gn and gn >= 0


@requires_cuda_bf16
def test_system_cuda_evaluate_respects_autocast_bfloat16(test_assets: dict[str, Path]) -> None:
    cfg = tiny_cfg(
        device="cuda",
        data_dir=str(test_assets["data_dir"]),
        data_format="bin",
        tokenizer=str(test_assets["tokenizer_json"]),
    )
    tok = tg.build_tokenizer(str(test_assets["tokenizer_json"]), text=None, tiktoken_encoding="gpt2")
    train_ds, val_ds, _, _ = tg.prepare_datasets(cfg, tok)
    _, val_loader, _ = tg.build_dataloaders(cfg, tg.torch.device("cuda"), train_ds, val_ds)
    model = tg.GPT(cfg, tok.vocab_size).to("cuda")
    val = tg.evaluate(
        model,
        val_loader,
        tg.torch.device("cuda"),
        use_amp=True,
        autocast_cuda_dtype=tg.torch.bfloat16,
        max_batches=2,
    )
    assert val == val
