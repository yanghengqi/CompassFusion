#!/usr/bin/env python3
"""Run and compare SPP constellation combinations against an IE ECEF truth file."""
from __future__ import annotations

import argparse
import csv
import os
import hashlib
import pickle
import time
from dataclasses import replace
from typing import Dict, Iterable, Tuple

import numpy as np

from compass.core.transforms import ecef2llh
from compass.gnss.spp import SPPSolver
from compass.io.rinex_native import RINEXNativeReader


def load_ie_truth(path: str) -> Dict[Tuple[int, int], np.ndarray]:
    """Read Inertial Explorer text: Week, GPSTime, X/Y/Z-ECEF, ..., Q."""
    truth: Dict[Tuple[int, int], np.ndarray] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            fields = line.split()
            if len(fields) < 6:
                continue
            try:
                week = int(round(float(fields[0])))
                sow = float(fields[1])
                xyz = np.asarray([float(fields[2]), float(fields[3]), float(fields[4])])
                quality = int(fields[-1])
            except ValueError:
                continue
            if quality == 1 and np.all(np.isfinite(xyz)):
                truth[(week, round(sow))] = xyz
    if not truth:
        raise ValueError(f"No quality-1 IE truth epochs found in {path}")
    return truth


def ecef_error_to_enu(error_xyz: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
    lat, lon = ecef2llh(reference_xyz)[:2]
    return np.array([
        [-np.sin(lon), np.cos(lon), 0.0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
    ]) @ error_xyz


def write_csv(path: str, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare SPP constellation combinations using IE truth")
    parser.add_argument("--obs", default=r"data\obs\KVH01960.21o")
    parser.add_argument("--nav", default=r"data\nav\brdm1960.21p")
    parser.add_argument("--gt", default=r"data\GT\2021196.txt")
    parser.add_argument("--systems", nargs="+", default=["G", "GE", "GRE", "GRCE", "R", "C", "E"])
    parser.add_argument("--elev", type=float, default=15.0, help="Elevation mask in degrees")
    parser.add_argument("--max-epochs", type=int, default=1100, help="0 means all observation epochs")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--out-dir", default=r"data\run_compare")
    parser.add_argument("--no-cache", action="store_true", help="Disable parsed RINEX cache")
    args = parser.parse_args(argv)

    combos = []
    for value in args.systems:
        combo = "".join(dict.fromkeys(value.upper()))
        if not combo or any(c not in "GRECIJ" for c in combo):
            parser.error(f"invalid constellation combination: {value}")
        if combo not in combos:
            combos.append(combo)

    os.makedirs(args.out_dir, exist_ok=True)
    truth = load_ie_truth(args.gt)
    reader = RINEXNativeReader()

    limit = max(0, args.max_epochs)
    signature_data = (
        "rinex-parser-v2",
        os.path.abspath(args.obs), os.path.getsize(args.obs), os.stat(args.obs).st_mtime_ns,
        os.path.abspath(args.nav), os.path.getsize(args.nav), os.stat(args.nav).st_mtime_ns,
        limit,
    )
    signature = hashlib.sha256(repr(signature_data).encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join(args.out_dir, ".cache")
    cache_path = os.path.join(cache_dir, f"rinex_{signature}.pkl")
    load_started = time.perf_counter()

    if not args.no_cache and os.path.isfile(cache_path):
        with open(cache_path, "rb") as stream:
            cached = pickle.load(stream)
        nav = cached["nav"]
        observations = cached["observations"]
        reader.nav_ion_gps = cached["ion_gps"]
        reader.nav_ion_bds = cached["ion_bds"]
        print(f"Loaded parsed RINEX cache in {time.perf_counter() - load_started:.2f} s")
    else:
        nav = reader.read_nav(args.nav)
        observations = reader.read_obs(args.obs, max_epochs=limit)
        print(f"Parsed RINEX files in {time.perf_counter() - load_started:.2f} s")
        if not args.no_cache:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as stream:
                pickle.dump({
                    "nav": nav,
                    "observations": observations,
                    "ion_gps": reader.nav_ion_gps,
                    "ion_bds": reader.nav_ion_bds,
                }, stream, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved parsed cache: {cache_path}")

    print(f"Observation epochs: {len(observations)}; IE truth epochs: {len(truth)}")
    print(f"Combinations: {', '.join(combos)}")
    summary_rows = []

    for combo in combos:
        allowed = set(combo)
        solver = SPPSolver(
            elev_mask=args.elev,
            ion_gps=reader.nav_ion_gps,
            ion_bds=reader.nav_ion_bds,
        )
        solution_rows = []
        error_rows = []
        successes = 0
        print(f"\n[{combo}] starting...", flush=True)

        for index, epoch in enumerate(observations, 1):
            selected = replace(
                epoch,
                observations=[ob for ob in epoch.observations if ob.system in allowed],
            )
            position, _clock, status = solver.solve(selected, nav, None)
            successes += int(status == 1)
            llh = ecef2llh(position) if status == 1 else np.full(3, np.nan)
            solution_rows.append([
                epoch.week, epoch.timestamp, *position,
                np.degrees(llh[0]), np.degrees(llh[1]), llh[2],
                status, len(selected.observations),
            ])

            key = (epoch.week, round(epoch.timestamp))
            if status == 1 and key in truth:
                dxyz = position - truth[key]
                enu = ecef_error_to_enu(dxyz, truth[key])
                error_rows.append([
                    epoch.week, epoch.timestamp, *dxyz, *enu,
                    np.hypot(enu[0], enu[1]), np.linalg.norm(dxyz),
                ])

            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(observations)):
                print(
                    f"[{combo}] {index}/{len(observations)} "
                    f"success={successes} matched_GT={len(error_rows)}",
                    flush=True,
                )

        solution_path = os.path.join(args.out_dir, f"spp_{combo}.csv")
        error_path = os.path.join(args.out_dir, f"spp_error_{combo}.csv")
        write_csv(
            solution_path,
            ["week", "sow", "x_m", "y_m", "z_m", "lat_deg", "lon_deg", "h_m", "status", "observed_sats"],
            solution_rows,
        )
        write_csv(
            error_path,
            ["week", "sow", "dx_m", "dy_m", "dz_m", "east_m", "north_m", "up_m", "horizontal_m", "error_3d_m"],
            error_rows,
        )

        if error_rows:
            values = np.asarray([row[5:] for row in error_rows], dtype=float)
            enu, horizontal, error_3d = values[:, :3], values[:, 3], values[:, 4]
            rmse_enu = np.sqrt(np.mean(enu * enu, axis=0))
            summary = [
                combo, len(observations), successes, len(error_rows),
                *rmse_enu, np.mean(enu[:, 2]),
                np.sqrt(np.mean(horizontal * horizontal)),
                np.sqrt(np.mean(error_3d * error_3d)),
                np.median(error_3d), np.percentile(error_3d, 95), np.max(error_3d),
            ]
        else:
            summary = [combo, len(observations), successes, 0, *([np.nan] * 9)]
        summary_rows.append(summary)
        print(
            f"[{combo}] done: success={successes}/{len(observations)}, "
            f"H-RMSE={summary[8]:.3f} m, 3D-RMSE={summary[9]:.3f} m",
            flush=True,
        )

    summary_path = os.path.join(args.out_dir, "spp_compare_summary.csv")
    write_csv(
        summary_path,
        [
            "systems", "epochs", "success", "gt_matched",
            "east_rmse_m", "north_rmse_m", "up_rmse_m", "up_mean_m",
            "horizontal_rmse_m", "error_3d_rmse_m", "error_3d_median_m",
            "error_3d_p95_m", "error_3d_max_m",
        ],
        summary_rows,
    )
    print(f"\nSummary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
