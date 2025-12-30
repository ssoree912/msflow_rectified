#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def collect_csvs(roots: Sequence[Path], pattern: str) -> List[Tuple[Path, Path]]:
    matches: List[Tuple[Path, Path]] = []
    for root in roots:
        if root.is_file():
            matches.append((root.parent, root))
            continue
        for path in root.rglob(pattern):
            if path.is_file():
                matches.append((root, path))
    return matches


def read_rows(paths: Iterable[Tuple[Path, Path]]) -> Tuple[List[str], List[Dict[str, str]]]:
    header: List[str] = []
    rows: List[Dict[str, str]] = []
    for root, path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in header:
                        header.append(name)
            for row in reader:
                row["source_root"] = str(root)
                row["source_path"] = str(path)
                rows.append(row)
    for name in ("source_root", "source_path"):
        if name not in header:
            header.append(name)
    return header, rows


def write_csv(header: Sequence[str], rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_output_path(roots: Sequence[Path], out_csv: Optional[str]) -> Path:
    if out_csv:
        return Path(out_csv)
    if len(roots) == 1:
        return roots[0] / "merged_metrics.csv"
    return Path.cwd() / "merged_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge best_metrics.csv files under one or more roots.")
    parser.add_argument("roots", nargs="+", help="One or more root directories (or files) to search.")
    parser.add_argument("--pattern", default="best_metrics.csv", help="Filename pattern to search for.")
    parser.add_argument("--out-csv", default="", help="Output CSV path.")
    args = parser.parse_args()

    roots = [Path(r) for r in args.roots]
    matched = collect_csvs(roots, args.pattern)
    if not matched:
        raise SystemExit(f"No CSV files matched pattern '{args.pattern}' in: {', '.join(map(str, roots))}")

    header, rows = read_rows(matched)
    out_path = resolve_output_path(roots, args.out_csv or None)
    write_csv(header, rows, out_path)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
