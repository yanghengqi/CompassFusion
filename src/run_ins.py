#!/usr/bin/env python3
"""Run ECEF INS mechanization or GNSS/INS loose coupling."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from compass.ins import (
    read_imu_file,
    read_gnss_pv_csv,
    read_inertial_explorer_pv,
    read_inertial_explorer_attitudes,
    initialize_state_from_gnss,
)
from compass.ins.coupling import LooselyCoupledINS, INSFilterConfig
from compass.ins.mechanization import ECEFInertialState, attitude_from_ie_hpr_rfu


def _parse_vector3(text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected numeric comma-separated values") from exc


def _nearest_attitude(attitudes, time: float, max_dt: float = 1.0):
    if not attitudes:
        return None
    nearest = min(attitudes, key=lambda item: abs(item.time - time))
    if abs(nearest.time - time) > max_dt:
        return None
    return nearest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu", required=True)
    parser.add_argument("--gnss", required=True, help="PPK/PPP CSV with ECEF position and velocity")
    parser.add_argument("--gnss-format", choices=("csv", "ie"), default="csv", help="GNSS input format")
    parser.add_argument("--imu-gyro-unit", choices=("rad/s", "rps", "dps", "deg/s"), default="rad/s")
    parser.add_argument("--imu-axis-order", default="rfu", help="Input IMU axis order; GREAT garfu/gafrd suffixes are supported")
    parser.add_argument("--init-attitude-ie", default="", help="Optional IE/GREAT truth file used only for initial heading/pitch/roll")
    parser.add_argument("--attitude-ie-mode", choices=("init", "epoch"), default="init", help="Use IE attitude once for initialization or at each GNSS epoch for diagnostics")
    parser.add_argument("--out", default=r"data\ins_result.csv")
    parser.add_argument("--mode", choices=("mechanization", "loose"), default="loose")
    parser.add_argument("--start-sow", type=float, default=0.0)
    parser.add_argument("--end-sow", type=float, default=0.0)
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--output-rate", type=float, default=1.0, help="Output rate in Hz; <=0 writes every IMU sample")
    parser.add_argument("--position-sigma", type=float, default=5.0)
    parser.add_argument("--velocity-sigma", type=float, default=2.0)
    parser.add_argument("--attitude-sigma-deg", type=float, default=20.0)
    parser.add_argument("--gate-sigma", type=float, default=1000.0)
    parser.add_argument("--lever-arm-body", type=_parse_vector3, default=(0.0, 0.0, 0.0), help="IMU-to-GNSS lever arm in body axes, meters: x,y,z")
    parser.add_argument("--estimate-attitude-bias", action="store_true", help="Enable attitude and IMU bias correction in the INS error-state filter")
    parser.add_argument("--velocity-attitude-aiding", action="store_true", help="Use GNSS velocity and local vertical to aid RFU attitude")
    parser.add_argument("--velocity-attitude-min-speed", type=float, default=2.0)
    parser.add_argument("--velocity-attitude-gain", type=float, default=1.0)
    args = parser.parse_args()

    if args.gnss_format == "ie":
        gnss = read_inertial_explorer_pv(args.gnss, fixed_only=args.fixed_only)
    else:
        gnss = read_gnss_pv_csv(args.gnss, fixed_only=args.fixed_only)
    if args.start_sow:
        gnss = [item for item in gnss if item.time >= args.start_sow]
    if args.end_sow:
        gnss = [item for item in gnss if item.time <= args.end_sow]
    if not gnss:
        raise SystemExit("No usable GNSS measurements")
    start = args.start_sow or gnss[0].time
    end = args.end_sow or gnss[-1].time
    imu = read_imu_file(args.imu, start_sow=start, end_sow=end, gyro_unit=args.imu_gyro_unit, axis_order=args.imu_axis_order)
    if not imu:
        raise SystemExit("No usable IMU samples")

    first = gnss[0]
    init_window = [sample for sample in imu if first.time <= sample.time <= first.time + 1.0]
    initial = initialize_state_from_gnss(first.time, first.position, first.velocity, init_window)
    attitudes = read_inertial_explorer_attitudes(args.init_attitude_ie) if args.init_attitude_ie else []
    if args.init_attitude_ie:
        attitude = _nearest_attitude(attitudes, first.time)
        if attitude is None:
            raise SystemExit("No IE attitude found near initial GNSS epoch")
        initial.C_be = attitude_from_ie_hpr_rfu(initial.position, attitude.heading_deg, attitude.pitch_deg, attitude.roll_deg)
    config = INSFilterConfig(
        initial_position_sigma_m=args.position_sigma,
        initial_velocity_sigma_mps=args.velocity_sigma,
        initial_attitude_sigma_rad=np.deg2rad(args.attitude_sigma_deg),
        min_position_sigma_m=0.05,
        min_velocity_sigma_mps=0.08,
        measurement_gate_sigma=args.gate_sigma,
        lever_arm_body_m=args.lever_arm_body,
        estimate_attitude_bias=args.estimate_attitude_bias,
        velocity_attitude_aiding=args.velocity_attitude_aiding,
        velocity_attitude_min_speed_mps=args.velocity_attitude_min_speed,
        velocity_attitude_gain=args.velocity_attitude_gain,
    )
    filt = LooselyCoupledINS(initial, config)
    gnss_index = 0
    output_period = 0.0 if args.output_rate <= 0.0 else 1.0 / args.output_rate
    next_output = start
    written = 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sow", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps",
            "sdx_m", "sdy_m", "sdz_m", "sdv_mps", "mode", "gnss_used", "innovation_norm",
            "gyro_bias_x", "gyro_bias_y", "gyro_bias_z", "accel_bias_x", "accel_bias_y", "accel_bias_z",
        ])
        for sample in imu:
            if sample.time < filt.state.time:
                continue
            snapshot = filt.propagate(sample)
            if args.mode == "loose":
                while gnss_index < len(gnss) and gnss[gnss_index].time <= sample.time + 0.5 * max(sample.dt, 0.0):
                    measurement = gnss[gnss_index]
                    if measurement.time >= filt.state.time - max(sample.dt, 0.0):
                        if args.attitude_ie_mode == "epoch":
                            attitude = _nearest_attitude(attitudes, measurement.time)
                            if attitude is not None:
                                filt.state.C_be = attitude_from_ie_hpr_rfu(filt.state.position, attitude.heading_deg, attitude.pitch_deg, attitude.roll_deg)
                        snapshot = filt.update_pv(measurement)
                    gnss_index += 1
            if output_period == 0.0 or sample.time + 1.0e-9 >= next_output:
                diag = np.sqrt(np.maximum(np.diag(snapshot.covariance), 0.0))
                writer.writerow([
                    f"{snapshot.time:.3f}",
                    *(f"{value:.4f}" for value in snapshot.position),
                    *(f"{value:.5f}" for value in snapshot.velocity),
                    *(f"{value:.4f}" for value in diag[:3]),
                    f"{float(np.mean(diag[3:6])):.4f}",
                    snapshot.mode,
                    snapshot.gnss_used,
                    f"{snapshot.innovation_norm:.3f}",
                    *(f"{value:.8g}" for value in filt.state.gyro_bias),
                    *(f"{value:.8g}" for value in filt.state.accel_bias),
                ])
                written += 1
                next_output += output_period if output_period > 0.0 else 0.0
    print(f"Finished INS {args.mode}: imu_samples={len(imu)} gnss={len(gnss)} rows={written} output={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
