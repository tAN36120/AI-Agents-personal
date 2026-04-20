#!/usr/bin/env python3
"""Compare Experiment 4 threshold sweep results without matplotlib dependency."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--datasets", default="Cars,SVHN")
    parser.add_argument("--retain-ratio", type=float, default=0.95)
    parser.add_argument("--report-json", default="./results/exp4_comparison_report.json")
    parser.add_argument("--report-csv", default="./results/exp4_comparison_report.csv")
    return parser.parse_args()


def load_records(results_dir: Path) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    for fpath in sorted(results_dir.glob("negations_negmerge_layer_zeroed_thresh_*.json")):
        m = re.search(r"thresh_([\-\d]+)\.json$", fpath.name)
        if not m:
            continue
        thresh = int(m.group(1))
        with fpath.open() as f:
            records[thresh] = json.load(f)
    return records


def choose_baseline(records: Dict[int, Dict[str, Any]]) -> int:
    if 0 in records:
        return 0
    if -1 in records:
        return -1
    return min(records.keys())


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    records = load_records(results_dir)
    if not records:
        raise FileNotFoundError(f"No threshold result files found in {results_dir}")

    baseline_t = choose_baseline(records)
    baseline = records[baseline_t]

    rows = []
    summary = {"baseline_threshold": baseline_t, "datasets": {}, "rows": []}

    for thresh in sorted(records.keys()):
        for ds in datasets:
            if ds not in records[thresh] or ds not in baseline:
                continue
            forget = float(records[thresh][ds]["test"])
            retain = float(records[thresh][ds]["test_control"])
            base_forget = float(baseline[ds]["test"])
            base_retain = float(baseline[ds]["test_control"])
            row = {
                "threshold": thresh,
                "dataset": ds,
                "forget": forget,
                "retain": retain,
                "delta_forget_vs_baseline": forget - base_forget,
                "delta_retain_vs_baseline": retain - base_retain,
                "retain_constraint": retain >= args.retain_ratio * base_retain,
            }
            rows.append(row)

    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds and r["threshold"] > 0 and r["retain_constraint"]]
        best = min(ds_rows, key=lambda x: x["forget"]) if ds_rows else None
        summary["datasets"][ds] = best

    report_json = Path(args.report_json).resolve()
    report_csv = Path(args.report_csv).resolve()
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)

    summary["rows"] = rows
    with report_json.open("w") as f:
        json.dump(summary, f, indent=2)

    with report_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "dataset",
                "forget",
                "retain",
                "delta_forget_vs_baseline",
                "delta_retain_vs_baseline",
                "retain_constraint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 90)
    print(f"Baseline threshold: {baseline_t}")
    print(f"Results dir: {results_dir}")
    print("=" * 90)
    for ds in datasets:
        best = summary["datasets"].get(ds)
        if not best:
            print(f"[{ds}] No threshold (>0) satisfies retain >= {args.retain_ratio:.2f} * baseline")
            continue
        print(
            f"[{ds}] best threshold={best['threshold']} "
            f"forget={best['forget']:.4f} retain={best['retain']:.4f} "
            f"Δforget={best['delta_forget_vs_baseline']:+.4f} "
            f"Δretain={best['delta_retain_vs_baseline']:+.4f}"
        )

    print("\nSaved:")
    print(f"  - {report_json}")
    print(f"  - {report_csv}")


if __name__ == "__main__":
    main()
