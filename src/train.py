"""Lesson 16: reproducible training pipeline (config, checkpoint, resume)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
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


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg, ROOT)
    return cfg


def validate_config(cfg: dict, root: Path) -> None:
    model = cfg["model"]
    train = cfg["training"]
    data = cfg["data"]
    if model["embed_dim"] % model["num_heads"] != 0:
        raise ValueError("embed_dim must be divisible by num_heads")
    for key, value in [
        ("batch_size", train["batch_size"]),
        ("learning_rate", train["learning_rate"]),
        ("num_steps", train["num_steps"]),
        ("eval_interval", train["eval_interval"]),
        ("eval_steps", train["eval_steps"]),
        ("checkpoint_interval", train["checkpoint_interval"]),
        ("gradient_clip", train["gradient_clip"]),
        ("block_size", model["block_size"]),
    ]:
        if value <= 0:
            raise ValueError(f"{key} must be > 0, got {value}")
    for key in ("train_path", "validation_path", "tokenizer_path"):
        path = root / data[key]
        if not path.exists():
            raise FileNotFoundError(f"missing {key}: {path}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


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


def setup_experiment(cfg: dict, config_src: Path) -> Path:
    exp_dir = ROOT / "experiments" / cfg["experiment_name"]
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = exp_dir / "config.yaml"
    if config_src.resolve() != dest.resolve():
        shutil.copy2(config_src, dest)
    return exp_dir


def setup_logging(exp_dir: Path) -> None:
    log_path = exp_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
    metrics: dict,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config,
        "metrics": metrics,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return int(ckpt["step"]), ckpt.get("metrics", {})


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    max_batches: int,
) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        B, T, V = logits.shape
        total += criterion(logits.view(B * T, V), y.view(B * T)).item()
        n += 1
        if n >= max_batches:
            break
    model.train()
    return total / max(n, 1)


def train_one_step(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
) -> float:
    optimizer.zero_grad()
    logits = model(x)
    B, T, V = logits.shape
    loss = criterion(logits.view(B * T, V), y.view(B * T))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach())


def append_metrics(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_loaders(cfg: dict, device_hint: str = "cpu"):
    del device_hint
    data = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    tokenizer = Tokenizer.from_file(str(ROOT / data["tokenizer_path"]))
    vocab = tokenizer.get_vocab_size()
    if vocab != model_cfg["vocab_size"]:
        raise ValueError(
            f"tokenizer vocab_size={vocab} != config model.vocab_size={model_cfg['vocab_size']}"
        )
    train_ids = texts_to_ids(tokenizer, load_texts(ROOT / data["train_path"]))
    val_ids = texts_to_ids(tokenizer, load_texts(ROOT / data["validation_path"]))
    train_ds = TokenDataset(train_ids, model_cfg["block_size"])
    val_ds = TokenDataset(val_ids, model_cfg["block_size"])
    if len(train_ds) == 0:
        raise RuntimeError("train TokenDataset is empty; lower block_size or grow corpus")
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True
    )
    val_loader = (
        DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)
        if len(val_ds) > 0
        else None
    )
    return tokenizer, train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson 16 training pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_dir = setup_experiment(cfg, config_path)
    setup_logging(exp_dir)
    metrics_path = exp_dir / "metrics.jsonl"
    ckpt_dir = exp_dir / "checkpoints"

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    gpt_config = GPTConfig(
        vocab_size=model_cfg["vocab_size"],
        block_size=model_cfg["block_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        embed_dim=model_cfg["embed_dim"],
        rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
        use_rope=bool(model_cfg.get("use_rope", True)),
    )
    model = GPTModel(gpt_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    _, train_loader, val_loader = build_loaders(cfg)
    train_iter = iter(train_loader)

    start_step = 0
    best_val = float("inf")
    bad_evals = 0
    last_train = float("nan")
    last_val = float("nan")

    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else ROOT / args.resume
        start_step, metrics = load_checkpoint(resume_path, model, optimizer)
        best_val = float(metrics.get("best_val_loss", best_val))
        logging.info("Resuming from step: %s", start_step)
        if start_step >= int(train_cfg["num_steps"]):
            logging.info(
                "Already at/above num_steps=%s; nothing to do. Raise num_steps to continue.",
                train_cfg["num_steps"],
            )
            return

    n_params = sum(p.numel() for p in model.parameters())
    logging.info("device=%s params=%s exp=%s", device, n_params, exp_dir)

    model.train()
    end_step = start_step
    for step in range(start_step + 1, int(train_cfg["num_steps"]) + 1):
        end_step = step
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)
        last_train = train_one_step(
            model,
            x,
            y,
            criterion,
            optimizer,
            float(train_cfg["gradient_clip"]),
        )

        do_eval = step % int(train_cfg["eval_interval"]) == 0 or step == 1
        if do_eval and val_loader is not None:
            last_val = estimate_loss(
                model,
                val_loader,
                criterion,
                device,
                int(train_cfg["eval_steps"]),
            )
            logging.info(
                "step=%s train_loss=%.4f val_loss=%.4f",
                step,
                last_train,
                last_val,
            )
            append_metrics(
                metrics_path,
                {"step": step, "train_loss": last_train, "val_loss": last_val},
            )
            if last_val < best_val:
                best_val = last_val
                bad_evals = 0
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model,
                    optimizer,
                    step,
                    cfg,
                    {
                        "best_val_loss": best_val,
                        "last_train_loss": last_train,
                        "last_val_loss": last_val,
                    },
                )
            else:
                bad_evals += 1
        elif step % 10 == 0:
            logging.info("step=%s train_loss=%.4f", step, last_train)
            append_metrics(
                metrics_path, {"step": step, "train_loss": last_train}
            )

        if step % int(train_cfg["checkpoint_interval"]) == 0:
            metrics = {
                "best_val_loss": best_val,
                "last_train_loss": last_train,
                "last_val_loss": last_val,
            }
            save_checkpoint(
                ckpt_dir / "latest.pt", model, optimizer, step, cfg, metrics
            )
            save_checkpoint(
                ckpt_dir / f"step-{step:04d}.pt",
                model,
                optimizer,
                step,
                cfg,
                metrics,
            )
            logging.info("saved checkpoint at step %s", step)

        es = train_cfg.get("early_stopping") or {}
        if es.get("enabled") and bad_evals >= int(es.get("patience", 5)):
            logging.info(
                "early stopping at step %s (patience=%s)", step, es["patience"]
            )
            break

    metrics = {
        "best_val_loss": best_val,
        "last_train_loss": last_train,
        "last_val_loss": last_val,
    }
    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, end_step, cfg, metrics)
    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, end_step, cfg, metrics)

    logging.info("Training finished")
    logging.info("-----------------")
    logging.info("start step: %s", start_step)
    logging.info("end step: %s", end_step)
    logging.info("best val loss: %s", best_val)
    logging.info("last train loss: %s", last_train)
    logging.info("last val loss: %s", last_val)
    logging.info("best checkpoint: %s", ckpt_dir / "best.pt")
    logging.info("latest checkpoint: %s", ckpt_dir / "latest.pt")


if __name__ == "__main__":
    main()
