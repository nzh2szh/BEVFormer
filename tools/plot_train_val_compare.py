#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib

# Use non-interactive backend for server/headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot train-vs-val curves from train_val_compare.tsv"
    )
    parser.add_argument(
        "--compare-file",
        required=True,
        help="path to train_val_compare.tsv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output image path; default: <compare-file-stem>.png",
    )
    parser.add_argument(
        "--train-loss-key",
        default="train_last_loss_align",
        choices=["train_last_loss_align", "train_avg_loss_align", "train_last_loss", "train_avg_loss"],
        help="optional extra train loss column to plot (in addition to default align losses)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="image dpi",
    )
    return parser.parse_args()


def to_float_or_none(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.upper() == "NA":
        return None
    try:
        return float(s)
    except Exception:
        return None


def load_rows(compare_file: Path):
    rows = []
    with compare_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            epoch = to_float_or_none(row.get("epoch"))
            if epoch is None:
                continue
            row["epoch"] = int(epoch)
            rows.append(row)
    rows.sort(key=lambda r: r["epoch"])
    return rows


def series(rows, key):
    xs = []
    ys = []
    for r in rows:
        y = to_float_or_none(r.get(key))
        if y is None:
            continue
        xs.append(r["epoch"])
        ys.append(y)
    return xs, ys


def main():
    args = parse_args()
    compare_file = Path(args.compare_file)
    if not compare_file.exists():
        raise FileNotFoundError(f"compare file not found: {compare_file}")

    output = Path(args.output) if args.output else compare_file.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(compare_file)
    if not rows:
        raise ValueError("No valid rows found in compare file.")

    # Loss panel
    x_tr_last_align, y_tr_last_align = series(rows, "train_last_loss_align")
    x_tr_avg_align, y_tr_avg_align = series(rows, "train_avg_loss_align")
    x_tr_loss, y_tr_loss = series(rows, args.train_loss_key)
    x_val_loss, y_val_loss = series(rows, "val_loss_align")

    # Top1 panel
    x_tr_i2t, y_tr_i2t = series(rows, "train_last_acc_i2t_top1")
    x_tr_t2i, y_tr_t2i = series(rows, "train_last_acc_t2i_top1")
    x_val_i2t, y_val_i2t = series(rows, "val_i2t_top1")
    x_val_t2i, y_val_t2i = series(rows, "val_t2i_top1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax0 = axes[0]
    if x_tr_last_align:
        ax0.plot(
            x_tr_last_align,
            y_tr_last_align,
            marker="o",
            linewidth=2,
            label="train_last_loss_align",
        )
    if x_tr_avg_align:
        ax0.plot(
            x_tr_avg_align,
            y_tr_avg_align,
            marker="o",
            linewidth=2,
            linestyle="--",
            label="train_avg_loss_align",
        )
    if args.train_loss_key not in ("train_last_loss_align", "train_avg_loss_align") and x_tr_loss:
        ax0.plot(x_tr_loss, y_tr_loss, marker="o", linewidth=2, label=args.train_loss_key)
    if x_val_loss:
        ax0.plot(x_val_loss, y_val_loss, marker="o", linewidth=2, label="val_loss_align")
    ax0.set_title("Loss Compare")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    ax1 = axes[1]
    if x_tr_i2t:
        ax1.plot(x_tr_i2t, y_tr_i2t, marker="o", linewidth=2, label="train_acc_i2t_top1")
    if x_tr_t2i:
        ax1.plot(x_tr_t2i, y_tr_t2i, marker="o", linewidth=2, label="train_acc_t2i_top1")
    if x_val_i2t:
        ax1.plot(x_val_i2t, y_val_i2t, marker="o", linewidth=2, label="val_i2t_top1")
    if x_val_t2i:
        ax1.plot(x_val_t2i, y_val_t2i, marker="o", linewidth=2, label="val_t2i_top1")
    ax1.set_title("Top1 Compare")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Top1")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    fig.suptitle(f"Train vs Val Metrics: {compare_file.name}")
    fig.savefig(output, dpi=args.dpi)
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
