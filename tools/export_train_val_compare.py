#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export epoch-aligned train/val comparison table from work_dir logs."
    )
    parser.add_argument("--work-dir", required=True, help="work directory path")
    parser.add_argument(
        "--train-log",
        default=None,
        help="optional single train log file path; default aggregates all top-level timestamp logs in work_dir",
    )
    parser.add_argument(
        "--val-summary",
        default="val_metrics.tsv",
        help="val summary file name relative to work_dir",
    )
    parser.add_argument(
        "--val-metrics-json-dir",
        default="val_metrics_json",
        help="per-epoch val metrics json dir relative to work_dir",
    )
    parser.add_argument(
        "--output",
        default="train_val_compare.tsv",
        help="output path. If relative, it is resolved under work_dir.",
    )
    return parser.parse_args()


def pick_latest_train_log(work_dir: Path):
    logs = sorted(
        [
            p
            for p in work_dir.glob("*.log")
            if p.is_file() and not p.name.startswith("epoch_")
        ],
        key=lambda p: p.stat().st_mtime,
    )
    if not logs:
        raise FileNotFoundError("No top-level training .log file found in work_dir")
    return logs[-1]


def list_train_logs(work_dir: Path):
    # Keep only timestamp-style training logs, skip val logs and json sidecars.
    ts_name = re.compile(r"^\d{8}_\d{6}$")
    logs = sorted(
        [
            p
            for p in work_dir.glob("*.log")
            if p.is_file() and ts_name.match(p.stem)
        ],
        key=lambda p: p.stat().st_mtime,
    )
    return logs


def _to_float(text):
    try:
        return float(text)
    except Exception:
        return None


def parse_train_log(train_log: Path):
    # Example:
    # Epoch [4][100/108] ... loss_align: 5.8549, acc_i2t_top1: 0.3400,
    # acc_t2i_top1: 0.2733, loss: 5.8549, grad_norm: 4.6993
    line_re = re.compile(
        r"Epoch\s*\[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?"
        r"loss_align:\s*(?P<loss_align>[-+0-9.eE]+).*?"
        r"acc_i2t_top1:\s*(?P<acc_i2t>[-+0-9.eE]+).*?"
        r"acc_t2i_top1:\s*(?P<acc_t2i>[-+0-9.eE]+).*?"
        r"loss:\s*(?P<loss>[-+0-9.eE]+)"
    )

    by_epoch = {}
    with train_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            epoch = int(m.group("epoch"))
            rec = {
                "iter": int(m.group("iter")),
                "iter_total": int(m.group("total")),
                "loss_align": _to_float(m.group("loss_align")),
                "acc_i2t_top1": _to_float(m.group("acc_i2t")),
                "acc_t2i_top1": _to_float(m.group("acc_t2i")),
                "loss": _to_float(m.group("loss")),
            }
            by_epoch.setdefault(epoch, []).append(rec)

    out = {}
    for epoch, rows in by_epoch.items():
        rows_sorted = sorted(rows, key=lambda x: x["iter"])
        last = rows_sorted[-1]
        out[epoch] = {
            "train_log_steps": len(rows_sorted),
            "train_last_iter": last["iter"],
            "train_iter_total": last["iter_total"],
            "train_last_loss_align": last["loss_align"],
            "train_last_acc_i2t_top1": last["acc_i2t_top1"],
            "train_last_acc_t2i_top1": last["acc_t2i_top1"],
            "train_last_loss": last["loss"],
            "train_avg_loss_align": mean([r["loss_align"] for r in rows_sorted if r["loss_align"] is not None]),
            "train_avg_acc_i2t_top1": mean([r["acc_i2t_top1"] for r in rows_sorted if r["acc_i2t_top1"] is not None]),
            "train_avg_acc_t2i_top1": mean([r["acc_t2i_top1"] for r in rows_sorted if r["acc_t2i_top1"] is not None]),
            "train_avg_loss": mean([r["loss"] for r in rows_sorted if r["loss"] is not None]),
        }
    return out


def parse_train_logs(train_logs):
    merged = {}
    for p in train_logs:
        part = parse_train_log(p)
        # If the same epoch appears in multiple logs, keep the later log's parse.
        merged.update(part)
    return merged


def parse_val_summary(val_summary: Path):
    out = {}
    if not val_summary.exists():
        return out
    with val_summary.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                epoch = int(row.get("epoch", ""))
            except Exception:
                continue
            out[epoch] = {
                "val_loss_align": row.get("val_loss_align", "NA"),
                "val_i2t_top1": row.get("i2t_top1", "NA"),
                "val_t2i_top1": row.get("t2i_top1", "NA"),
                "val_i2t_r5": row.get("i2t_r5", "NA"),
                "val_i2t_r10": row.get("i2t_r10", "NA"),
                "val_t2i_r5": row.get("t2i_r5", "NA"),
                "val_t2i_r10": row.get("t2i_r10", "NA"),
                "val_log": row.get("val_log", ""),
                "load_report": row.get("load_report", ""),
            }
    return out


def parse_val_metrics_json_dir(metrics_dir: Path):
    out = {}
    if not metrics_dir.exists() or not metrics_dir.is_dir():
        return out
    for p in sorted(metrics_dir.glob("epoch_*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        epoch = data.get("epoch", None)
        if isinstance(epoch, int):
            out[epoch] = {
                "val_return_code": data.get("val_return_code", "NA"),
                "val_metrics_json": str(p),
            }
    return out


def fmt_num(x):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    train_log = Path(args.train_log) if args.train_log else None
    if train_log is not None and not train_log.exists():
        raise FileNotFoundError(f"train log not found: {train_log}")

    val_summary = work_dir / args.val_summary
    metrics_json_dir = work_dir / args.val_metrics_json_dir

    if train_log is not None:
        train_logs = [train_log]
    else:
        train_logs = list_train_logs(work_dir)
        if not train_logs:
            # Backward-compatible fallback for unusual naming.
            train_logs = [pick_latest_train_log(work_dir)]

    train_by_epoch = parse_train_logs(train_logs)
    val_by_epoch = parse_val_summary(val_summary)
    val_json_by_epoch = parse_val_metrics_json_dir(metrics_json_dir)

    all_epochs = sorted(set(train_by_epoch.keys()) | set(val_by_epoch.keys()) | set(val_json_by_epoch.keys()))

    out_arg = Path(args.output)
    out_path = out_arg if out_arg.is_absolute() else (work_dir / out_arg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "epoch",
        "train_log_steps",
        "train_last_iter",
        "train_iter_total",
        "train_last_loss_align",
        "train_last_acc_i2t_top1",
        "train_last_acc_t2i_top1",
        "train_last_loss",
        "train_avg_loss_align",
        "train_avg_acc_i2t_top1",
        "train_avg_acc_t2i_top1",
        "train_avg_loss",
        "val_loss_align",
        "val_i2t_top1",
        "val_t2i_top1",
        "val_i2t_r5",
        "val_i2t_r10",
        "val_t2i_r5",
        "val_t2i_r10",
        "val_return_code",
        "val_log",
        "load_report",
        "val_metrics_json",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for epoch in all_epochs:
            row = {"epoch": epoch}

            tr = train_by_epoch.get(epoch, {})
            for k in [
                "train_log_steps",
                "train_last_iter",
                "train_iter_total",
                "train_last_loss_align",
                "train_last_acc_i2t_top1",
                "train_last_acc_t2i_top1",
                "train_last_loss",
                "train_avg_loss_align",
                "train_avg_acc_i2t_top1",
                "train_avg_acc_t2i_top1",
                "train_avg_loss",
            ]:
                row[k] = fmt_num(tr.get(k, "NA"))

            va = val_by_epoch.get(epoch, {})
            for k in [
                "val_loss_align",
                "val_i2t_top1",
                "val_t2i_top1",
                "val_i2t_r5",
                "val_i2t_r10",
                "val_t2i_r5",
                "val_t2i_r10",
                "val_log",
                "load_report",
            ]:
                row[k] = va.get(k, "NA")

            vj = val_json_by_epoch.get(epoch, {})
            row["val_return_code"] = vj.get("val_return_code", "NA")
            row["val_metrics_json"] = vj.get("val_metrics_json", "NA")

            writer.writerow(row)

    if len(train_logs) == 1:
        print(f"train_log: {train_logs[0]}")
    else:
        print(f"train_logs: {len(train_logs)} files (latest: {train_logs[-1]})")
    print(f"val_summary: {val_summary}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
