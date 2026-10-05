"""Lesson 18: compare FP32 / FP16 / BF16 training on the same tiny GPT workload."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpt import GPTConfig, GPTModel  # noqa: E402
from train import build_loaders, load_config, set_seed  # noqa: E402


PRECISION_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def mode_supported(precision: str, device: str) -> tuple[bool, str]:
    if precision == "fp32":
        return True, ""
    if device == "cuda":
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            return False, "CUDA device does not support BF16"
        return True, ""
    # CPU: autocast for fp16/bf16 exists in recent PyTorch; still may be slow.
    return True, ""


def check_finite_params(model: nn.Module) -> bool:
    for p in model.parameters():
        if not torch.isfinite(p).all():
            return False
    return True


def probe_dtypes(model: nn.Module, x: torch.Tensor, device: str, dtype: torch.dtype) -> None:
    model.eval()
    print("dtype probe (illustrative; AMP rules may keep some ops in higher precision)")
    print(f"input x: {x.dtype}")
    with torch.autocast(device_type=device, dtype=dtype):
        tok = model.token_embedding(x)  # слой представлений токенов
        print(f"token_embedding out: {tok.dtype}")
        h = model.blocks[0].ln1(tok)  # Pre-Norm перед вниманием
        print(f"block0 ln1 out: {h.dtype}")
        logits = model(x)
        print(f"logits: {logits.dtype}")
        B, T, V = logits.shape
        y = x  # для зонда достаточно той же формы; CE только ради dtype
        loss = nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))
        print(f"loss: {loss.dtype}")
    model.train()


def train_loop(
    *,
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    precision: str,
    num_steps: int,
    grad_clip: float,
    use_scaler: bool,
) -> dict:
    model.train()
    it = iter(loader)
    loss_history: list[float] = []
    skipped_updates = 0
    finite = True
    final_loss = float("nan")

    amp_dtype = PRECISION_DTYPE[precision]
    use_amp = precision != "fp32"
    scaler = None
    if use_amp and precision == "fp16" and use_scaler:
        scaler = torch.amp.GradScaler(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for step in range(1, num_steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device, dtype=amp_dtype):
                logits = model(x)
                B, T, V = logits.shape
                loss = criterion(logits.view(B * T, V), y.view(B * T))
        else:
            logits = model(x)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))

        if not torch.isfinite(loss):
            finite = False
            final_loss = float(loss.detach()) if loss.numel() == 1 else float("nan")
            loss_history.append(final_loss)
            print(f"step={step} NON-FINITE loss — stop")
            break

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped_updates += 1
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if not check_finite_params(model):
            finite = False
            final_loss = float(loss.detach())
            loss_history.append(final_loss)
            print(f"step={step} NON-FINITE parameters — stop")
            break

        final_loss = float(loss.detach())
        loss_history.append(final_loss)
        if step == 1 or step % 5 == 0 or step == num_steps:
            print(f"step={step:3d}  loss={final_loss:.4f}")

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mb = None
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    steps_done = len(loss_history)
    return {
        "precision": precision,
        "device": device,
        "steps": steps_done,
        "time_sec": round(elapsed, 4),
        "steps_per_sec": round(steps_done / elapsed, 4) if elapsed > 0 else None,
        "peak_memory_mb": None if peak_mb is None else round(peak_mb, 2),
        "final_loss": final_loss,
        "finite": finite and check_finite_params(model),
        "skipped_scaler_updates": skipped_updates,
        "use_scaler": scaler is not None,
        "loss_history": [round(v, 6) if math.isfinite(v) else v for v in loss_history],
        "supported": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lesson 18 mixed-precision benchmark")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        required=True,
    )
    parser.add_argument(
        "--no-scaler",
        action="store_true",
        help="FP16 without GradScaler (demo; may be unstable)",
    )
    parser.add_argument(
        "--probe-dtypes",
        action="store_true",
        help="print a few intermediate dtypes under autocast and exit",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    device = pick_device()
    precision = args.precision

    out_dir = ROOT / "experiments" / "precision"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{precision}.json"

    ok, reason = mode_supported(precision, device)
    print(f"Precision benchmark: {precision}")
    print(f"device={device}  torch={torch.__version__}")
    if not ok:
        payload = {
            "precision": precision,
            "device": device,
            "supported": False,
            "reason": reason,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"unsupported: {reason}")
        print(f"saved {out_path}")
        return

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
    _, train_loader, _ = build_loaders(cfg)

    if args.probe_dtypes:
        if precision == "fp32":
            print("probe-dtypes is for fp16/bf16 autocast; use --precision fp16 or bf16")
            return
        x, _ = next(iter(train_loader))
        x = x.to(device)
        probe_dtypes(model, x, device, PRECISION_DTYPE[precision])
        return

    use_scaler = precision == "fp16" and not args.no_scaler
    print(
        f"mode={precision}  steps={train_cfg['num_steps']}  "
        f"scaler={'on' if use_scaler else 'off'}"
    )
    result = train_loop(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        precision=precision,
        num_steps=int(train_cfg["num_steps"]),
        grad_clip=float(train_cfg["gradient_clip"]),
        use_scaler=use_scaler,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("---")
    print(
        f"time_sec={result['time_sec']}  steps/sec={result['steps_per_sec']}  "
        f"peak_memory_mb={result['peak_memory_mb']}  "
        f"final_loss={result['final_loss']:.4f}  finite={result['finite']}"
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
