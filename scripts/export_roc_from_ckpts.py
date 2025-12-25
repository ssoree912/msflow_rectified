#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def collect_csv_paths(root: Path, pattern: str) -> list[Path]:
    return sorted((p for p in root.rglob(pattern) if p.is_file()), key=lambda p: str(p))


def read_rows(paths: list[Path]) -> tuple[list[str] | None, list[dict]]:
    header = None
    rows = []
    for path in paths:
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                continue
            if header is None:
                header = reader.fieldnames
            elif reader.fieldnames != header:
                raise ValueError(f"CSV header mismatch in {path}")
            for row in reader:
                if not any(row.values()):
                    continue
                rows.append(row)
    return header, rows


def main():
    parser = argparse.ArgumentParser(description="Merge per-class metrics CSV files under one or more root directories.")
    parser.add_argument("roots", nargs="+", help="Root directories to search for CSV files.")
    parser.add_argument("--pattern", default="best_metrics.csv", help="Glob pattern for CSV files.")
    parser.add_argument("--out-csv", default="", help="Output CSV path.")
    args = parser.parse_args()

    roots = [Path(root).expanduser() for root in args.roots]
    for root in roots:
        if not root.exists():
            raise SystemExit(f"Root path does not exist: {root}")

    csv_paths = []
    for root in roots:
        csv_paths.extend(collect_csv_paths(root, args.pattern))
    if not csv_paths:
        raise SystemExit(f"No CSV files found under roots with pattern '{args.pattern}'")

    header, rows = read_rows(csv_paths)
    if header is None:
        raise SystemExit("No valid CSV headers found.")

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser()
    else:
        out_path = roots[0] / "merged_metrics.csv" if len(roots) == 1 else Path("merged_metrics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows from {len(csv_paths)} files to {out_path}")


if __name__ == "__main__":
    main()
