#!/usr/bin/env python3
"""Compare PPP result for POTS against IGS SINEX coordinates (DOY 196)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from compass.core.transforms import ecef2llh

ROOT = Path(__file__).resolve().parents[1]

# igs21P21664.snx estimate at GPS week 2179 day 196
SNX_XYZ = np.array([3800689.375, 882077.650, 5028791.485])


def read_ppp(path: Path) -> list[tuple[float, np.ndarray]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "1":
                continue
            sow = float(row["sow"])
            xyz = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
            rows.append((sow, xyz))
    return rows


def main(argv=None) -> int:
    ppp_path = ROOT / "data" / "ppp_result_POTS1960.csv"
    if len(argv or sys.argv) > 1:
        ppp_path = Path(argv[1] if argv else sys.argv[1])
    if not ppp_path.is_file():
        print(f"Missing {ppp_path}; run scripts/run_pots_196.ps1 first")
        return 1

    rows = read_ppp(ppp_path)
    if not rows:
        print("No successful PPP epochs in result file")
        return 1

    llh = ecef2llh(SNX_XYZ)
    lat, lon = llh[:2]
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    ecef_to_enu = np.array([
        [-sl, cl, 0.0],
        [-sp * cl, -sp * sl, cp],
        [cp * cl, cp * sl, sp],
    ])
    sow = np.array([t for t, _ in rows])
    enu = (ecef_to_enu @ np.array([xyz - SNX_XYZ for _, xyz in rows]).T).T
    error_3d = np.linalg.norm(enu, axis=1)

    print(f"POTS vs IGS SINEX ({len(rows)} epochs)")
    for label, mask in (
        ("all", np.ones(len(rows), dtype=bool)),
        ("after 15 min", sow >= sow[0] + 900.0),
    ):
        rms = np.sqrt(np.mean(enu[mask] ** 2, axis=0))
        rms_3d = np.sqrt(np.mean(error_3d[mask] ** 2))
        p95 = np.percentile(error_3d[mask], 95)
        print(
            f"  {label}: 3D RMS={rms_3d:.3f} m  p95={p95:.3f} m  "
            f"ENU RMS={rms[0]:.3f}/{rms[1]:.3f}/{rms[2]:.3f} m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
