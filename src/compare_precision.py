"""Lesson 18: summarize experiments/precision/*.json into a table and optional plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "experiments" / "precision"
ORDER = ("fp32", "fp16", "bf16")


def load_results(directory: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in ORDER:
        path = directory / f"{name}.json"
        if not path.exists():
            continue
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def print_table(results: dict[str, dict]) -> None:
    print(
        f"{'Mode':<6}  {'OK':<5}  {'Time_s':>8}  {'steps/s':>8}  "
        f"{'PeakMiB':>8}  {'FinalLoss':>10}  {'Finite':<6}"
    )
    print("-" * 72)
    for name in ORDER:
        row = results.get(name)
        if row is None:
            print(f"{name:<6}  {'—':<5}  {'missing':>8}")
            continue
        if not row.get("supported", True):
            print(f"{name:<6}  {'no':<5}  unsupported: {row.get('reason', '')}")
            continue
        peak = row.get("peak_memory_mb")
        peak_s = "n/a" if peak is None else f"{peak:.1f}"
        print(
            f"{name:<6}  {'yes':<5}  {row.get('time_sec', float('nan')):8.3f}  "
            f"{row.get('steps_per_sec', float('nan')):8.3f}  "
            f"{peak_s:>8}  {row.get('final_loss', float('nan')):10.4f}  "
            f"{str(row.get('finite')):<6}"
        )


def maybe_plot(results: dict[str, dict], directory: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skip plots")
        return

    modes = []
    mem = []
    sps = []
    for name in ORDER:
        row = results.get(name)
        if not row or not row.get("supported", True):
            continue
        modes.append(name.upper())
        mem.append(row.get("peak_memory_mb"))
        sps.append(row.get("steps_per_sec"))

    if not modes:
        return

    # Memory chart (skip if all null — typical on CPU)
    if any(v is not None for v in mem):
        ys = [0.0 if v is None else float(v) for v in mem]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(modes, ys, color=["#4C78A8", "#F58518", "#54A24B"][: len(modes)])
        ax.set_ylabel("peak GPU memory (MiB)")
        ax.set_title("Precision benchmark — peak memory")
        fig.tight_layout()
        path = directory / "peak_memory.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print("saved", path)
    else:
        print("peak memory: all n/a (CPU run) — skip memory chart")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(modes, sps, color=["#4C78A8", "#F58518", "#54A24B"][: len(modes)])
    ax.set_ylabel("steps/sec")
    ax.set_title("Precision benchmark — throughput")
    fig.tight_layout()
    path = directory / "steps_per_sec.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", path)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name in ORDER:
        row = results.get(name)
        if not row or not row.get("supported", True):
            continue
        hist = row.get("loss_history") or []
        if hist:
            ax.plot(range(1, len(hist) + 1), hist, label=name.upper())
    ax.set_xlabel("step")
    ax.set_ylabel("train loss")
    ax.set_title("Precision benchmark — loss curves")
    ax.legend()
    fig.tight_layout()
    path = directory / "loss_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare precision benchmark JSON files")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    directory = args.dir if args.dir.is_absolute() else ROOT / args.dir
    results = load_results(directory)
    if not results:
        raise SystemExit(f"no JSON results in {directory}")
    print_table(results)
    if not args.no_plot:
        maybe_plot(results, directory)


if __name__ == "__main__":
    main()
