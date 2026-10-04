"""Lesson 17: diagnose broken / unstable training without changing the trainer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt import GPTConfig, GPTModel  # noqa: E402


class TokenDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx: int):
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def load_texts(path: Path) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            texts.append(json.loads(line)["text"])
    return texts


def texts_to_ids(tokenizer: Tokenizer, texts: list[str]) -> list[int]:
    eos_id = tokenizer.token_to_id("[EOS]")
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
        if eos_id is not None:
            ids.append(eos_id)
    return ids


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: str) -> GPTModel:
    m = cfg["model"]
    config = GPTConfig(
        vocab_size=m["vocab_size"],
        block_size=m["block_size"],
        num_layers=m["num_layers"],
        num_heads=m["num_heads"],
        embed_dim=m["embed_dim"],
        rope_theta=float(m.get("rope_theta", 10000.0)),
        use_rope=bool(m.get("use_rope", True)),
    )
    return GPTModel(config).to(device)


def build_loader(cfg: dict) -> DataLoader:
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["training"]
    tokenizer = Tokenizer.from_file(str(ROOT / data["tokenizer_path"]))
    if tokenizer.get_vocab_size() != model["vocab_size"]:
        raise ValueError("tokenizer vocab_size != config model.vocab_size")
    ids = texts_to_ids(tokenizer, load_texts(ROOT / data["train_path"]))
    ds = TokenDataset(ids, model["block_size"])
    if len(ds) == 0:
        raise RuntimeError("empty TokenDataset")
    return DataLoader(ds, batch_size=train["batch_size"], shuffle=True)


def check_finite_tensor(name: str, t: torch.Tensor) -> tuple[str, str]:
    if not torch.isfinite(t).all():
        return "FAILED", f"{name}: non-finite values detected"
    return "OK", f"{name}: finite"


def check_finite_model(model: nn.Module) -> tuple[str, str]:
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            return "FAILED", f"NON-FINITE PARAMETER:\n{name}"
    return "OK", "all parameters finite"


def parameter_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        total += float(p.detach().float().norm().item() ** 2)
    return math.sqrt(total)


def gradient_norm(model: nn.Module) -> float:
    return float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
    )


def print_section(title: str, status: str, detail: str = "") -> None:
    print(title)
    if detail:
        print(f"{status}: {detail}")
    else:
        print(status)
    print()


def diagnose_once(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: str,
) -> dict[str, str]:
    results: dict[str, str] = {}
    x, y = batch
    x, y = x.to(device), y.to(device)

    status, detail = check_finite_tensor("input x", x.float())
    # token ids are integers — finiteness via cast is always true; check range instead
    if (x < 0).any():
        status, detail = "FAILED", "input x: negative token ids"
    results["DATA"] = status
    print_section("DATA", status, detail if status != "OK" else "")

    expected_t = model.config.block_size
    if x.ndim != 2 or y.shape != x.shape:
        results["INPUT SHAPES"] = "FAILED"
        print_section("INPUT SHAPES", "FAILED", f"got x={tuple(x.shape)} y={tuple(y.shape)}")
    elif x.size(1) > expected_t:
        results["INPUT SHAPES"] = "FAILED"
        print_section(
            "INPUT SHAPES",
            "FAILED",
            f"T={x.size(1)} > block_size={expected_t}",
        )
    else:
        results["INPUT SHAPES"] = "OK"
        print_section("INPUT SHAPES", "OK", f"x={tuple(x.shape)} y={tuple(y.shape)}")

    model.train()
    try:
        logits = model(x)
        results["FORWARD"] = "OK"
        print_section("FORWARD", "OK")
    except Exception as exc:  # noqa: BLE001 — diagnostic surface
        results["FORWARD"] = "FAILED"
        print_section("FORWARD", "FAILED", str(exc))
        return results

    st, detail = check_finite_tensor("logits", logits)
    results["LOGITS"] = st
    print_section("LOGITS", st, "" if st == "OK" else detail)

    B, T, V = logits.shape
    loss = criterion(logits.view(B * T, V), y.view(B * T))
    if not torch.isfinite(loss):
        results["LOSS"] = "FAILED"
        print_section("LOSS", "FAILED", "non-finite loss detected")
        return results
    results["LOSS"] = "OK"
    print_section("LOSS", "OK", f"value={float(loss.detach()):.4f}")

    model.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = gradient_norm(model)
    if not math.isfinite(gnorm):
        results["GRADIENTS"] = "FAILED"
        print_section("GRADIENTS", "FAILED", "non-finite gradient norm")
    elif gnorm > 100.0:
        results["GRADIENTS"] = "WARNING"
        print_section("GRADIENTS", "WARNING", f"large gradient norm={gnorm:.4f}")
    else:
        results["GRADIENTS"] = "OK"
        print_section("GRADIENTS", "OK", f"grad_norm={gnorm:.4f}")

    st, detail = check_finite_model(model)
    pnorm = parameter_norm(model)
    if st == "OK":
        print_section("PARAMETERS", "OK", f"param_norm={pnorm:.4f}")
    else:
        print_section("PARAMETERS", st, detail)
    results["PARAMETERS"] = st

    if device.startswith("cuda") and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print_section(
            "GPU MEMORY",
            "OK",
            f"allocated={alloc:.1f} MiB reserved={reserved:.1f} MiB",
        )
        results["GPU MEMORY"] = "OK"
    else:
        print_section("GPU MEMORY", "OK", "CUDA not available — skipped (CPU run)")
        results["GPU MEMORY"] = "OK"

    return results


def run_demo(cfg: dict, device: str, mode: str) -> None:
    model = build_model(cfg, device)
    loader = build_loader(cfg)
    criterion = nn.CrossEntropyLoss()
    lr = 3.0 if mode == "high-lr" else float(cfg["training"]["learning_rate"])
    clip = float(cfg["training"]["gradient_clip"]) if mode == "ok" else 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"Demo mode: {mode}  lr={lr}  clip={clip}")
    print("step   loss     grad_norm   param_norm")
    it = iter(loader)
    for step in range(1, 6):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), y.view(B * T))
        if not torch.isfinite(loss):
            print(f"{step:4d}  LOSS NON-FINITE — stop demo")
            break
        loss.backward()
        gnorm = gradient_norm(model)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        pnorm = parameter_norm(model)
        print(
            f"{step:4d}  {float(loss.detach()):7.4f}  {gnorm:9.4f}  {pnorm:10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson 17 training diagnostics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--demo",
        choices=("ok", "high-lr"),
        default=None,
        help="short training demo: healthy vs high learning rate",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = load_config(config_path)
    torch.manual_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training diagnostics")
    print("--------------------")
    print(f"device={device}")
    print()

    if args.demo is not None:
        run_demo(cfg, device, args.demo)
        return

    model = build_model(cfg, device)
    if args.checkpoint is not None:
        ckpt_path = (
            args.checkpoint
            if args.checkpoint.is_absolute()
            else ROOT / args.checkpoint
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"loaded checkpoint: {ckpt_path} (step={ckpt.get('step', '?')})")
        print()

    loader = build_loader(cfg)
    batch = next(iter(loader))
    criterion = nn.CrossEntropyLoss()
    diagnose_once(model, batch, criterion, device)


if __name__ == "__main__":
    main()
