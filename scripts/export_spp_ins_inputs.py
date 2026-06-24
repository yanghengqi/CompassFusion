#!/usr/bin/env python3
"""Export SPP P/V and range measurements from real RINEX for INS coupling."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.core.constants import CLIGHT
from compass.gnss.ionosphere import ionocorr
from compass.gnss.spp import SPPSolver
from compass.io.rinex_native import RINEXNativeReader

_SYSTEMS = frozenset("GRECRJ")


def _systems(value: str) -> frozenset[str]:
    letters = frozenset(ch.upper() for ch in value if ch.isalpha())
    bad = letters - _SYSTEMS
    if bad:
        raise argparse.ArgumentTypeError(f"unsupported systems: {''.join(sorted(bad))}")
    return letters or frozenset("GEC")


def _finite_position(position: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(position)) and np.linalg.norm(position) > 1.0e6)


def _filter_epoch(epoch, systems: frozenset[str]):
    epoch.observations = [obs for obs in epoch.observations if obs.system in systems]
    return epoch


def _sat_ranges(solver: SPPSolver, epoch, nav, user_pos: np.ndarray, code_sigma_m: float) -> list[list[str]]:
    rows: list[list[str]] = []
    obs_gps_time = epoch.week * 604800.0 + epoch.timestamp
    user_llh = None
    try:
        from compass.core.transforms import ecef2llh

        user_llh = ecef2llh(user_pos)
    except Exception:
        user_llh = None
    for sat_obs in epoch.observations:
        if sat_obs.system not in {"G", "E", "C", "R", "J"}:
            continue
        if sat_obs.system == "C" and sat_obs.sat_id in solver.exclude_bds_prns:
            continue
        pr_raw = float(solver._get_pseudorange(sat_obs))
        if pr_raw <= 0.0 or not np.isfinite(pr_raw):
            continue
        sat_state = solver._cached_sat_state(sat_obs, obs_gps_time, nav, pr_raw)
        if sat_state is None:
            continue
        nav_record, sat_pos, sat_vel, sat_clk = sat_state
        if sat_pos is None or sat_clk is None:
            continue
        if user_llh is not None:
            azel = solver._compute_azel(sat_pos, user_pos, user_llh)
            if azel[1] < solver.elev_mask:
                continue
            sagnac = solver.OMGE * (sat_pos[0] * user_pos[1] - sat_pos[1] * user_pos[0]) / CLIGHT
            dion, _ = ionocorr(
                epoch.timestamp,
                solver._ion_gps,
                solver._ion_bds,
                user_llh,
                np.array([azel[0], azel[1]]),
                sat_obs.system,
            )
            freq = solver._freq_L1_hz_for_obs(sat_obs, nav_record)
            if freq > 0.0:
                dion *= (1575.42e6 / freq) ** 2
            trop = solver._tropospheric_delay(user_llh, azel[1])
        else:
            sagnac = 0.0
            dion = 0.0
            trop = 0.0
        pr = solver._apply_code_bias(pr_raw, nav_record, sat_obs.system, code_pr1=sat_obs.code_pr1)
        # The tight filter estimates receiver clock in metres; move satellite
        # clock to the observation side so predicted=rho+receiver_clock.
        corrected_pr = float(pr + CLIGHT * sat_clk - sagnac - dion - trop)
        if not np.isfinite(corrected_pr):
            continue
        row = [
            f"{epoch.timestamp:.3f}",
            sat_obs.system,
            sat_obs.sat_id,
            f"{sat_pos[0]:.4f}",
            f"{sat_pos[1]:.4f}",
            f"{sat_pos[2]:.4f}",
            f"{corrected_pr:.4f}",
            f"{code_sigma_m * code_sigma_m:.4f}",
        ]
        if sat_vel is not None and np.all(np.isfinite(sat_vel)):
            row.extend([f"{sat_vel[0]:.5f}", f"{sat_vel[1]:.5f}", f"{sat_vel[2]:.5f}"])
        else:
            row.extend(["", "", ""])
        row.extend(["", ""])
        rows.append(row)
    return rows


def _velocity_rows(pv_rows: list[dict[str, object]]) -> list[list[object]]:
    if not pv_rows:
        return []
    positions = [np.asarray(row["position"], dtype=float) for row in pv_rows]
    times = [float(row["sow"]) for row in pv_rows]
    velocities: list[np.ndarray] = []
    for index, position in enumerate(positions):
        if len(positions) == 1:
            velocity = np.zeros(3)
        elif index == 0:
            dt = max(times[1] - times[0], 1.0)
            velocity = (positions[1] - position) / dt
        elif index == len(positions) - 1:
            dt = max(times[-1] - times[-2], 1.0)
            velocity = (position - positions[-2]) / dt
        else:
            dt = max(times[index + 1] - times[index - 1], 1.0)
            velocity = (positions[index + 1] - positions[index - 1]) / dt
        velocities.append(velocity)
    rows = []
    for row, velocity in zip(pv_rows, velocities):
        pos = np.asarray(row["position"], dtype=float)
        rows.append(
            [
                row["week"],
                f"{float(row['sow']):.3f}",
                f"{pos[0]:.4f}",
                f"{pos[1]:.4f}",
                f"{pos[2]:.4f}",
                f"{velocity[0]:.5f}",
                f"{velocity[1]:.5f}",
                f"{velocity[2]:.5f}",
                "3.0000",
                "3.0000",
                "3.0000",
                1,
                0,
                row["ns"],
            ]
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", required=True)
    parser.add_argument("--nav", required=True)
    parser.add_argument("--pv-out", required=True)
    parser.add_argument("--ranges-out", required=True)
    parser.add_argument("--start-sow", type=float, default=0.0)
    parser.add_argument("--end-sow", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--elev", type=float, default=10.0)
    parser.add_argument("--systems", type=_systems, default=frozenset("GEC"))
    parser.add_argument("--code-sigma", type=float, default=3.0)
    parser.add_argument("--max-position-step", type=float, default=80.0, help="reject SPP epochs that jump this far from the last accepted epoch")
    args = parser.parse_args()

    reader = RINEXNativeReader()
    nav = reader.read_nav(args.nav)
    obs_epochs = reader.read_obs(args.obs, max_epochs=max(0, args.max_epochs), start_sow=args.start_sow)
    if args.end_sow:
        obs_epochs = [epoch for epoch in obs_epochs if epoch.timestamp <= args.end_sow]
    if args.max_epochs:
        obs_epochs = obs_epochs[: args.max_epochs]

    solver = SPPSolver(elev_mask=args.elev, ion_gps=reader.nav_ion_gps, ion_bds=reader.nav_ion_bds)
    approx = None
    last_good = None
    Path(args.pv_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ranges_out).parent.mkdir(parents=True, exist_ok=True)

    pv_rows: list[dict[str, object]] = []
    with Path(args.ranges_out).open("w", newline="", encoding="utf-8") as range_stream:
        range_writer = csv.writer(range_stream)
        range_writer.writerow(
            [
                "sow",
                "system",
                "sat_id",
                "sat_x_m",
                "sat_y_m",
                "sat_z_m",
                "pseudorange_m",
                "variance_m2",
                "sat_vx_mps",
                "sat_vy_mps",
                "sat_vz_mps",
                "range_rate_mps",
                "range_rate_variance_m2",
            ]
        )
        ok_count = 0
        range_count = 0
        rejected_qc = 0
        for epoch in obs_epochs:
            epoch = _filter_epoch(epoch, args.systems)
            if not epoch.observations:
                continue
            seed = approx if approx is not None else epoch.approx_position
            pos, _clk, status = solver.solve(epoch, nav, seed)
            if status and _finite_position(pos):
                if last_good is not None and args.max_position_step > 0.0:
                    step = float(np.linalg.norm(pos - last_good))
                    if step > args.max_position_step:
                        approx = last_good.copy()
                        rejected_qc += 1
                        continue
                approx = pos.copy()
                last_good = pos.copy()
                ok_count += 1
                pv_rows.append({"week": epoch.week, "sow": epoch.timestamp, "position": pos.copy(), "ns": len(epoch.observations)})
                rows = _sat_ranges(solver, epoch, nav, pos, args.code_sigma)
                range_writer.writerows(rows)
                range_count += len(rows)
            elif last_good is not None:
                approx = last_good.copy()

    with Path(args.pv_out).open("w", newline="", encoding="utf-8") as pv_stream:
        pv_writer = csv.writer(pv_stream)
        pv_writer.writerow(["week", "sow", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "sdx_m", "sdy_m", "sdz_m", "status", "fix_status", "ns"])
        pv_writer.writerows(_velocity_rows(pv_rows))

    print(
        f"Exported SPP/INS inputs: epochs={len(obs_epochs)} spp_ok={ok_count} "
        f"qc_rejected={rejected_qc} ranges={range_count} pv={args.pv_out} ranges_out={args.ranges_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
