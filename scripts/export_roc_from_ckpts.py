#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SOURCE_ROOT_COL = "source_root"
SOURCE_PATH_COL = "source_path"


def collect_csv_paths(root: Path, pattern: str) -> List[Path]:
    return sorted((p for p in root.rglob(pattern) if p.is_file()), key=lambda p: str(p))


def read_rows(entries: List[Tuple[Path, Path]]) -> Tuple[Optional[List[str]], List[Dict[str, str]]]:
    header = None
    rows = []
    for path, root in entries:
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
                row[SOURCE_ROOT_COL] = str(root)
                row[SOURCE_PATH_COL] = str(path)
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

    csv_entries: List[Tuple[Path, Path]] = []
    for root in roots:
        for path in collect_csv_paths(root, args.pattern):
            csv_entries.append((path, root))
    if not csv_entries:
        raise SystemExit(f"No CSV files found under roots with pattern '{args.pattern}'")

    header, rows = read_rows(csv_entries)
    if header is None:
        raise SystemExit("No valid CSV headers found.")
    out_header = list(header)
    for extra in (SOURCE_ROOT_COL, SOURCE_PATH_COL):
        if extra not in out_header:
            out_header.append(extra)

    if args.out_csv:
        out_path = Path(args.out_csv).expanduser()
    else:
        out_path = roots[0] / "merged_metrics.csv" if len(roots) == 1 else Path("merged_metrics.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows from {len(csv_entries)} files to {out_path}")


if __name__ == "__main__":
    main()
