"""Lesson 16: plot train/val loss from metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    steps_train, train_losses = [], []
    steps_val, val_losses = [], []
    with args.metrics.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "train_loss" in row:
                steps_train.append(row["step"])
                train_losses.append(row["train_loss"])
            if "val_loss" in row:
                steps_val.append(row["step"])
                val_losses.append(row["val_loss"])

    plt.figure(figsize=(8, 4.5))
    if steps_train:
        plt.plot(steps_train, train_losses, label="train_loss")
    if steps_val:
        plt.plot(steps_val, val_losses, label="val_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print("saved", args.out)


if __name__ == "__main__":
    main()
