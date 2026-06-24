#!/usr/bin/env python3
"""Compare PPP result against MSF_20211013 GNSS ground truth."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_gt(path: Path) -> dict[float, tuple[float, float, float]]:
    rows = {}
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("#") or "GPSTime" in line or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                sow = float(parts[1])
                x, y, z = map(float, parts[2:5])
            except ValueError:
                continue
            rows[sow] = (x, y, z)
    return rows


def read_ppp(path: Path) -> dict[float, tuple[float, float, float]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "1":
                continue
            sow = float(row["sow"])
            rows[sow] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
    return rows


def enu(dx: float, dy: float, dz: float, lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    e = -so * dx + co * dy
    n = -sl * co * dx - sl * so * dy + cl * dz
    u = cl * co * dx + cl * so * dy + sl * dz
    return float(e), float(n), float(u)


def main() -> int:
    gt_path = ROOT / "data" / "GT" / "groundtruth_211013_GNSS.txt"
    ppp_path = ROOT / "data" / "ppp_result_SEPT2860.csv"
    if len(sys.argv) > 1:
        ppp_path = Path(sys.argv[1])
    gt = read_gt(gt_path)
    ppp = read_ppp(ppp_path)
    errs = []
    for sow, (px, py, pz) in ppp.items():
        if sow not in gt:
            continue
        gx, gy, gz = gt[sow]
        lat = np.degrees(np.arctan2(gz, np.sqrt(gx * gx + gy * gy)))
        lon = np.degrees(np.arctan2(gy, gx))
        errs.append(enu(px - gx, py - gy, pz - gz, lat, lon))
    if not errs:
        print("no matched epochs")
        return 1
    arr = np.asarray(errs)
    hor = np.hypot(arr[:, 0], arr[:, 1])
    error_3d = np.linalg.norm(arr, axis=1)
    print(f"matched={len(arr)} / ppp={len(ppp)} / gt={len(gt)}")
    print(f"hor RMS={np.sqrt(np.mean(hor**2)):.3f} m  mean={np.mean(hor):.3f} m")
    print(f"E mean={np.mean(arr[:,0]):.3f}  N mean={np.mean(arr[:,1]):.3f}  U mean={np.mean(arr[:,2]):.3f} m")
    print(f"3D RMS={np.sqrt(np.mean(np.sum(arr**2, axis=1))):.3f} m")
    print(
        "3D P50/P95/P99/MAX="
        f"{np.quantile(error_3d, 0.50):.3f}/"
        f"{np.quantile(error_3d, 0.95):.3f}/"
        f"{np.quantile(error_3d, 0.99):.3f}/"
        f"{np.max(error_3d):.3f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
