#!/usr/bin/env python3
"""Compare INS CSV output against a GREAT/Inertial Explorer truth file."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.ins.coupling import read_inertial_explorer_pv


def read_ins_csv(path: Path) -> dict[int, np.ndarray]:
    rows: dict[int, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                sow = round(float(row["sow"]))
                rows[sow] = np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
            except (KeyError, ValueError):
                continue
    return rows


def summarize(result_path: Path, truth_path: Path) -> tuple[int, int, dict[str, float]]:
    result = read_ins_csv(result_path)
    truth = {round(item.time): item.position for item in read_inertial_explorer_pv(truth_path)}
    errors = []
    for sow, position in result.items():
        if sow in truth:
            errors.append(float(np.linalg.norm(position - truth[sow])))
    if not errors:
        return 0, len(result), {}
    arr = np.asarray(errors, dtype=float)
    stats = {
        "median": float(np.median(arr)),
        "rms": float(math.sqrt(np.mean(arr * arr))),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }
    return len(arr), len(result), stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("data/great_msf/MSF_20211013/groundtruth/groundtruth_211013_ADIS.txt"),
    )
    args = parser.parse_args()

    matched, total, stats = summarize(args.result, args.truth)
    if not stats:
        print(f"matched=0 / result={total}")
        return 1
    print(f"matched={matched} / result={total}")
    print(
        "3D median/RMS/P95/P99/MAX="
        f"{stats['median']:.4f}/"
        f"{stats['rms']:.4f}/"
        f"{stats['p95']:.4f}/"
        f"{stats['p99']:.4f}/"
        f"{stats['max']:.4f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
