#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TSV file to aligned Markdown table.")
    parser.add_argument("--input", required=True, help="input TSV path")
    parser.add_argument("--output", required=True, help="output Markdown path")
    parser.add_argument(
        "--columns",
        default="",
        help="comma-separated subset columns (optional), e.g. epoch,train_last_loss_align,val_loss_align",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise FileNotFoundError(f"input tsv not found: {in_path}")

    with in_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        all_cols = reader.fieldnames or []

    if not all_cols:
        raise ValueError("No header columns found in TSV.")

    if args.columns.strip():
        cols = [c.strip() for c in args.columns.split(",") if c.strip()]
        missing = [c for c in cols if c not in all_cols]
        if missing:
            raise ValueError(f"Columns not found in TSV: {missing}")
    else:
        cols = all_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute display widths for aligned markdown source readability.
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            v = str(r.get(c, ""))
            if len(v) > widths[c]:
                widths[c] = len(v)

    def fmt_row(items):
        return "| " + " | ".join(items) + " |"

    with out_path.open("w", encoding="utf-8") as f:
        header_cells = [c.ljust(widths[c]) for c in cols]
        f.write(fmt_row(header_cells) + "\n")
        sep_cells = ["-" * max(3, widths[c]) for c in cols]
        f.write(fmt_row(sep_cells) + "\n")
        for r in rows:
            cells = [str(r.get(c, "")).ljust(widths[c]) for c in cols]
            f.write(fmt_row(cells) + "\n")

    print(f"Saved markdown table: {out_path}")


if __name__ == "__main__":
    main()
