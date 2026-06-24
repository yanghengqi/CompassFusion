"""Loose and tight GNSS/INS coupling filters."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .mechanization import (
    ECEFInertialState,
    IMURecord,
    OMEGA_IE_MAT,
    initialize_state_from_gnss,
    attitude_from_velocity_rfu,
    normal_gravity_ecef,
    skew,
    strapdown_step_ecef,
)


@dataclass
class GNSSPVMeasurement:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    position_std: np.ndarray
    velocity_std: np.ndarray
    status: int = 1
    fix_status: int = 0

    @property
    def covariance(self) -> np.ndarray:
        covariance = np.zeros((6, 6), dtype=float)
        covariance[:3, :3] = np.diag(np.maximum(self.position_std, 1.0e-3) ** 2)
        covariance[3:6, 3:6] = np.diag(np.maximum(self.velocity_std, 1.0e-3) ** 2)
        return covariance


@dataclass
class RangeMeasurement:
    time: float
    sat_position: np.ndarray
    pseudorange: float
    variance: float
    sat_velocity: Optional[np.ndarray] = None
    range_rate: Optional[float] = None
    range_rate_variance: Optional[float] = None
    system: str = "G"


@dataclass
class IEAttitudeMeasurement:
    time: float
    heading_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass
class INSFilterConfig:
    gyro_noise_rad_s: float = 5.0e-4
    accel_noise_mps2: float = 2.0e-2
    gyro_bias_rw_rad_s2: float = 1.0e-6
    accel_bias_rw_mps3: float = 1.0e-4
    initial_position_sigma_m: float = 5.0
    initial_velocity_sigma_mps: float = 2.0
    initial_attitude_sigma_rad: float = math.radians(20.0)
    initial_gyro_bias_sigma_rad_s: float = 1.0e-2
    initial_accel_bias_sigma_mps2: float = 5.0e-1
    min_position_sigma_m: float = 0.02
    min_velocity_sigma_mps: float = 0.02
    measurement_gate_sigma: float = 12.0
    estimate_attitude_bias: bool = False
    hard_reset_pv: bool = True
    lever_arm_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_attitude_aiding: bool = False
    velocity_attitude_min_speed_mps: float = 2.0
    velocity_attitude_gain: float = 1.0
    initial_clock_bias_sigma_m: float = 1.0e6
    initial_clock_drift_sigma_mps: float = 1.0e3
    tight_range_gate_sigma: float = 15.0
    tight_min_ranges: int = 4
    tight_auto_init_clock: bool = True
    tight_position_ls_blend: float = 0.0
    initial_isb_sigma_m: float = 1.0e5
    isb_rw_mps: float = 0.01


@dataclass
class INSSnapshot:
    time: float
    position: np.ndarray
    velocity: np.ndarray
    C_be: np.ndarray
    covariance: np.ndarray
    mode: str
    gnss_used: int = 0
    innovation_norm: float = math.nan


def read_gnss_pv_csv(path: str | Path, fixed_only: bool = False) -> list[GNSSPVMeasurement]:
    """Read PPK/PPP CSV output as ECEF position/velocity measurements."""

    measurements: list[GNSSPVMeasurement] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                status = int(float(row.get("status", "1")))
                fix_status = int(float(row.get("fix_status", "0")))
                if status <= 0 or (fixed_only and fix_status <= 0):
                    continue
                time = float(row["sow"])
                position = np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
                velocity = np.asarray([row.get("vx_mps", "0"), row.get("vy_mps", "0"), row.get("vz_mps", "0")], dtype=float)
                std = np.asarray([row.get("sdx_m", "0.05"), row.get("sdy_m", "0.05"), row.get("sdz_m", "0.05")], dtype=float)
            except (KeyError, ValueError):
                continue
            velocity_std = np.full(3, 0.08 if fix_status else 0.5, dtype=float)
            measurements.append(GNSSPVMeasurement(time, position, velocity, std, velocity_std, status, fix_status))
    return measurements


def _csv_float(row: dict[str, str], *names: str, default: float | None = None) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    if default is not None:
        return default
    raise KeyError(names[0])


def read_range_measurements_csv(path: str | Path) -> dict[float, list[RangeMeasurement]]:
    """Read satellite range measurements for tightly coupled updates.

    Required columns are time, satellite ECEF position, pseudorange and
    variance. Accepted aliases:
    - time: sow or time
    - position: sat_x_m/sat_y_m/sat_z_m or sx_m/sy_m/sz_m
    - pseudorange: pseudorange_m or pr_m
    - variance: variance_m2 or pr_variance_m2

    Optional Doppler/range-rate columns are sat_vx_mps/sat_vy_mps/sat_vz_mps,
    range_rate_mps and range_rate_variance_m2.
    """

    epochs: dict[float, list[RangeMeasurement]] = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                time = _csv_float(row, "sow", "time")
                sat_position = np.asarray(
                    [
                        _csv_float(row, "sat_x_m", "sx_m", "x_m"),
                        _csv_float(row, "sat_y_m", "sy_m", "y_m"),
                        _csv_float(row, "sat_z_m", "sz_m", "z_m"),
                    ],
                    dtype=float,
                )
                pseudorange = _csv_float(row, "pseudorange_m", "pr_m")
                variance = _csv_float(row, "variance_m2", "pr_variance_m2", default=9.0)
                if not np.isfinite(pseudorange) or variance <= 0.0:
                    continue
                sat_velocity = None
                range_rate = None
                range_rate_variance = None
                if all(row.get(name, "") != "" for name in ("sat_vx_mps", "sat_vy_mps", "sat_vz_mps")):
                    sat_velocity = np.asarray(
                        [
                            _csv_float(row, "sat_vx_mps"),
                            _csv_float(row, "sat_vy_mps"),
                            _csv_float(row, "sat_vz_mps"),
                        ],
                        dtype=float,
                    )
                if row.get("range_rate_mps", "") != "":
                    range_rate = _csv_float(row, "range_rate_mps")
                    range_rate_variance = _csv_float(row, "range_rate_variance_m2", default=0.04)
            except (KeyError, ValueError):
                continue
            epochs.setdefault(time, []).append(
                RangeMeasurement(
                    time,
                    sat_position,
                    pseudorange,
                    variance,
                    sat_velocity,
                    range_rate,
                    range_rate_variance,
                    row.get("system", "G").strip().upper()[:1] or "G",
                )
            )
    return dict(sorted(epochs.items()))


def read_inertial_explorer_pv(path: str | Path, fixed_only: bool = False) -> list[GNSSPVMeasurement]:
    """Read GREAT/Inertial Explorer text truth files as ECEF position/velocity measurements."""

    measurements: list[GNSSPVMeasurement] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < 20:
                continue
            try:
                float(parts[0])
                time = float(parts[1])
                position = np.asarray(parts[2:5], dtype=float)
                velocity = np.asarray(parts[17:20], dtype=float)
            except ValueError:
                continue
            fix_status = 1 if "Fixed" in parts else 0
            if fixed_only and not fix_status:
                continue
            position_std = np.full(3, 0.05 if fix_status else 0.5, dtype=float)
            velocity_std = np.full(3, 0.05 if fix_status else 0.3, dtype=float)
            measurements.append(GNSSPVMeasurement(time, position, velocity, position_std, velocity_std, 1, fix_status))
    return measurements


def read_inertial_explorer_attitudes(path: str | Path) -> list[IEAttitudeMeasurement]:
    """Read Heading/Pitch/Roll columns from GREAT/Inertial Explorer text files."""

    attitudes: list[IEAttitudeMeasurement] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < 27:
                continue
            try:
                time = float(parts[1])
                heading = float(parts[24])
                pitch = float(parts[25])
                roll = float(parts[26])
            except ValueError:
                continue
            attitudes.append(IEAttitudeMeasurement(time, heading, pitch, roll))
    return attitudes


class LooselyCoupledINS:
    """15-state ECEF error-state GNSS/INS filter."""

    def __init__(self, initial: ECEFInertialState, config: INSFilterConfig | None = None, covariance: np.ndarray | None = None):
        self.state = initial.copy()
        self.config = config or INSFilterConfig()
        self.lever_arm_body = np.asarray(self.config.lever_arm_body_m, dtype=float).reshape(3)
        if covariance is None:
            sig = np.array(
                [self.config.initial_position_sigma_m] * 3
                + [self.config.initial_velocity_sigma_mps] * 3
                + [self.config.initial_attitude_sigma_rad] * 3
                + [self.config.initial_accel_bias_sigma_mps2] * 3
                + [self.config.initial_gyro_bias_sigma_rad_s] * 3,
                dtype=float,
            )
            self.P = np.diag(sig * sig)
        else:
            self.P = np.asarray(covariance, dtype=float).copy()

    def propagate(self, sample: IMURecord) -> INSSnapshot:
        dt = float(sample.dt)
        force_b = np.asarray(sample.accel, dtype=float) - self.state.accel_bias
        omega_b = np.asarray(sample.gyro, dtype=float) - self.state.gyro_bias
        C_be = self.state.C_be.copy()
        self.state = strapdown_step_ecef(self.state, sample)
        if dt > 0.0:
            F = np.zeros((15, 15), dtype=float)
            F[0:3, 3:6] = np.eye(3)
            F[3:6, 3:6] = -2.0 * OMEGA_IE_MAT
            F[3:6, 6:9] = -C_be @ skew(force_b)
            F[3:6, 9:12] = -C_be
            F[6:9, 6:9] = -skew(omega_b)
            F[6:9, 12:15] = -np.eye(3)
            Phi = np.eye(15) + F * dt
            q = np.array(
                [0.0] * 3
                + [self.config.accel_noise_mps2] * 3
                + [self.config.gyro_noise_rad_s] * 3
                + [self.config.accel_bias_rw_mps3] * 3
                + [self.config.gyro_bias_rw_rad_s2] * 3,
                dtype=float,
            )
            Q = np.diag((q * max(dt, 1.0e-6)) ** 2)
            self.P = Phi @ self.P @ Phi.T + Q
            self.P = (self.P + self.P.T) / 2.0
        return self.snapshot("mechanization")

    def update_pv(self, measurement: GNSSPVMeasurement) -> INSSnapshot:
        if self.config.velocity_attitude_aiding:
            self._aid_attitude_from_velocity(measurement)
        H = np.zeros((6, 15), dtype=float)
        H[:3, :3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        lever_ecef = self.state.C_be @ self.lever_arm_body
        H[:3, 6:9] = -skew(lever_ecef)
        predicted_position = self.state.position + lever_ecef
        z = np.r_[measurement.position - predicted_position, measurement.velocity - self.state.velocity]
        R = measurement.covariance.copy()
        floor_p = self.config.min_position_sigma_m ** 2
        floor_v = self.config.min_velocity_sigma_mps ** 2
        for i in range(3):
            R[i, i] = max(R[i, i], floor_p)
            R[i + 3, i + 3] = max(R[i + 3, i + 3], floor_v)
        snapshot = self._kalman_update(H, z, R, "loose", gnss_used=1)
        if self.config.hard_reset_pv and snapshot.gnss_used:
            self.state.position = measurement.position.copy() - self.state.C_be @ self.lever_arm_body
            self.state.velocity = measurement.velocity.copy()
            self.P[:3, :] = 0.0
            self.P[:, :3] = 0.0
            self.P[3:6, :] = 0.0
            self.P[:, 3:6] = 0.0
            self.P[:3, :3] = R[:3, :3]
            self.P[3:6, 3:6] = R[3:6, 3:6]
            snapshot = self.snapshot("loose", 1, snapshot.innovation_norm)
        return snapshot

    def _aid_attitude_from_velocity(self, measurement: GNSSPVMeasurement) -> None:
        speed = float(np.linalg.norm(measurement.velocity))
        if speed < self.config.velocity_attitude_min_speed_mps:
            return
        target = attitude_from_velocity_rfu(self.state.position, measurement.velocity)
        gain = min(max(float(self.config.velocity_attitude_gain), 0.0), 1.0)
        if gain >= 1.0:
            self.state.C_be = target
        elif gain > 0.0:
            self.state.C_be = _orthonormalize_local((1.0 - gain) * self.state.C_be + gain * target)

    def _kalman_update(self, H: np.ndarray, innovation: np.ndarray, R: np.ndarray, mode: str, gnss_used: int) -> INSSnapshot:
        S = H @ self.P @ H.T + R
        try:
            Sinv_innov = np.linalg.solve(S, innovation)
            nis = float(innovation @ Sinv_innov)
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            return self.snapshot(mode, 0, math.nan)
        gate = self.config.measurement_gate_sigma
        if innovation.size and nis > (gate * gate * innovation.size):
            return self.snapshot(mode, 0, math.sqrt(nis))
        dx = K @ innovation
        self._inject(dx)
        A = np.eye(self.P.shape[0]) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2.0
        return self.snapshot(mode, gnss_used, math.sqrt(max(nis, 0.0)))

    def _inject(self, dx: np.ndarray) -> None:
        self.state.position += dx[0:3]
        self.state.velocity += dx[3:6]
        if not self.config.estimate_attitude_bias:
            return
        dtheta = dx[6:9]
        self.state.C_be = _orthonormalize_local((np.eye(3) + skew(dtheta)) @ self.state.C_be)
        self.state.accel_bias += dx[9:12]
        self.state.gyro_bias += dx[12:15]

    def snapshot(self, mode: str, gnss_used: int = 0, innovation_norm: float = math.nan) -> INSSnapshot:
        return INSSnapshot(
            self.state.time,
            self.state.position.copy(),
            self.state.velocity.copy(),
            self.state.C_be.copy(),
            self.P.copy(),
            mode,
            gnss_used,
            innovation_norm,
        )


class TightlyCoupledINS(LooselyCoupledINS):
    """ECEF error-state INS with receiver clock states for raw GNSS updates.

    The state extends the 15 INS error states with receiver clock bias (m) and
    drift (m/s). Feed it satellite positions and pseudoranges from any GNSS
    front-end; the filter handles the INS-side update.
    """

    def __init__(self, initial: ECEFInertialState, config: INSFilterConfig | None = None, covariance: np.ndarray | None = None):
        super().__init__(initial, config, covariance)
        self.IDX_CLK = 15
        self.IDX_CLK_DRIFT = 16
        self.IDX_ISB = {"E": 17, "C": 18, "R": 19}
        self.state_dim = 20
        self.clock_bias_m = 0.0
        self.clock_drift_mps = 0.0
        self.isb_m = np.zeros(3, dtype=float)
        self.clock_initialized = False
        P = np.zeros((self.state_dim, self.state_dim), dtype=float)
        P[:15, :15] = self.P
        P[self.IDX_CLK, self.IDX_CLK] = self.config.initial_clock_bias_sigma_m ** 2
        P[self.IDX_CLK_DRIFT, self.IDX_CLK_DRIFT] = self.config.initial_clock_drift_sigma_mps ** 2
        for idx in self.IDX_ISB.values():
            P[idx, idx] = self.config.initial_isb_sigma_m ** 2
        self.P = P

    def _isb_index(self, system: str) -> int | None:
        return self.IDX_ISB.get((system or "G").upper()[:1])

    def _isb_value(self, system: str) -> float:
        idx = self._isb_index(system)
        if idx is None:
            return 0.0
        return float(self.isb_m[idx - 17])

    def propagate(self, sample: IMURecord) -> INSSnapshot:
        P15 = self.P[:15, :15].copy()
        tail = self.P[15:, 15:].copy()
        cross = self.P[:15, 15:].copy()
        self.P = P15
        snap = super().propagate(sample)
        dt = max(float(sample.dt), 0.0)
        P = np.zeros((self.state_dim, self.state_dim), dtype=float)
        P[:15, :15] = self.P
        Ftail = np.eye(5)
        Ftail[0, 1] = dt
        q = np.array([0.5 * dt, 0.05 * dt, self.config.isb_rw_mps * dt, self.config.isb_rw_mps * dt, self.config.isb_rw_mps * dt])
        P[15:, 15:] = Ftail @ tail @ Ftail.T + np.diag(q * q)
        P[:15, 15:] = cross
        P[15:, :15] = cross.T
        self.P = (P + P.T) / 2.0
        self.clock_bias_m += self.clock_drift_mps * dt
        snap.covariance = self.P.copy()
        return snap

    def _range_ls_position(self, measurements: list[RangeMeasurement]) -> tuple[np.ndarray, float, float] | None:
        usable = []
        for measurement in measurements:
            sat = np.asarray(measurement.sat_position, dtype=float).reshape(3)
            if float(np.linalg.norm(sat)) > 1.0e6 and np.isfinite(measurement.pseudorange):
                usable.append((sat, float(measurement.pseudorange), max(float(measurement.variance), 1.0), measurement.system))
        if len(usable) < self.config.tight_min_ranges:
            return None
        pos = self.state.position.copy()
        clk = float(self.clock_bias_m)
        systems = sorted({system for *_rest, system in usable if self._isb_index(system) is not None})
        sys_col = {system: 4 + index for index, system in enumerate(systems)}
        isb = {system: self._isb_value(system) for system in systems}
        for _ in range(6):
            rows = []
            residuals = []
            weights = []
            for sat, pr, variance, system in usable:
                dr = pos - sat
                rho = float(np.linalg.norm(dr))
                if rho <= 1.0:
                    continue
                line = dr / rho
                row = np.zeros(4 + len(systems), dtype=float)
                row[:4] = [line[0], line[1], line[2], 1.0]
                if system in sys_col:
                    row[sys_col[system]] = 1.0
                rows.append(row)
                residuals.append(pr - (rho + clk + isb.get(system, 0.0)))
                weights.append(1.0 / variance)
            if len(rows) < self.config.tight_min_ranges:
                return None
            H = np.asarray(rows, dtype=float)
            z = np.asarray(residuals, dtype=float)
            w = np.sqrt(np.asarray(weights, dtype=float))
            gate = max(float(self.config.tight_range_gate_sigma), 0.0)
            if gate > 0.0 and z.size > self.config.tight_min_ranges:
                normalized = np.abs(z - float(np.median(z))) * w
                keep = normalized <= gate
                if int(np.sum(keep)) >= self.config.tight_min_ranges:
                    H = H[keep]
                    z = z[keep]
                    w = w[keep]
            try:
                dx, *_ = np.linalg.lstsq(H * w[:, None], z * w, rcond=None)
            except np.linalg.LinAlgError:
                return None
            pos += dx[:3]
            clk += dx[3]
            for system, col in sys_col.items():
                isb[system] += dx[col]
            if float(np.linalg.norm(dx[:3])) < 1.0e-3 and abs(float(dx[3])) < 1.0e-3:
                break
        for system, value in isb.items():
            idx = self._isb_index(system)
            if idx is not None:
                self.isb_m[idx - 17] = value
        residual_norm = float(np.sqrt(np.mean(np.asarray(residuals, dtype=float) ** 2))) if residuals else math.nan
        return pos, clk, residual_norm

    def _range_rows(self, measurements: Iterable[RangeMeasurement]) -> tuple[list[np.ndarray], list[float], list[float]]:
        rows = []
        innovations = []
        variances = []
        for measurement in measurements:
            sat = np.asarray(measurement.sat_position, dtype=float).reshape(3)
            dr = self.state.position - sat
            rho = float(np.linalg.norm(dr))
            if rho <= 1.0:
                continue
            line = dr / rho
            row = np.zeros(self.state_dim, dtype=float)
            row[:3] = line
            row[self.IDX_CLK] = 1.0
            isb_idx = self._isb_index(measurement.system)
            if isb_idx is not None:
                row[isb_idx] = 1.0
            predicted = rho + self.clock_bias_m + self._isb_value(measurement.system)
            rows.append(row)
            innovations.append(float(measurement.pseudorange) - predicted)
            variances.append(max(float(measurement.variance), 1.0))
            if measurement.sat_velocity is not None and measurement.range_rate is not None:
                sat_velocity = np.asarray(measurement.sat_velocity, dtype=float).reshape(3)
                relative_velocity = self.state.velocity - sat_velocity
                predicted_rate = float(line @ relative_velocity + self.clock_drift_mps)
                row_rate = np.zeros(self.state_dim, dtype=float)
                row_rate[3:6] = line
                row_rate[self.IDX_CLK_DRIFT] = 1.0
                rows.append(row_rate)
                innovations.append(float(measurement.range_rate) - predicted_rate)
                variances.append(max(float(measurement.range_rate_variance or 1.0), 1.0e-4))
        return rows, innovations, variances

    def update_ranges(self, measurements: Iterable[RangeMeasurement]) -> INSSnapshot:
        measurement_list = list(measurements)
        blend = min(max(float(self.config.tight_position_ls_blend), 0.0), 1.0)
        if blend > 0.0:
            ls = self._range_ls_position(measurement_list)
            if ls is not None:
                ls_pos, ls_clock, ls_norm = ls
                self.state.position = (1.0 - blend) * self.state.position + blend * ls_pos
                self.clock_bias_m = (1.0 - blend) * self.clock_bias_m + blend * ls_clock
                return self.snapshot("tight", len(measurement_list), ls_norm)
        if self.config.tight_auto_init_clock and not self.clock_initialized:
            residuals = []
            for measurement in measurement_list:
                sat = np.asarray(measurement.sat_position, dtype=float).reshape(3)
                rho = float(np.linalg.norm(self.state.position - sat))
                if rho > 1.0:
                    residuals.append(float(measurement.pseudorange) - rho)
            if residuals:
                self.clock_bias_m = float(np.median(residuals))
                self.clock_initialized = True

        rows, innovations, variances = self._range_rows(measurement_list)
        if len(rows) < self.config.tight_min_ranges:
            return self.snapshot("tight", 0, math.nan)
        H = np.vstack(rows)
        R = np.diag(variances)
        z = np.asarray(innovations, dtype=float)

        gate = max(float(self.config.tight_range_gate_sigma), 0.0)
        if gate > 0.0 and z.size >= self.config.tight_min_ranges:
            for _ in range(3):
                scale = np.sqrt(np.maximum(np.diag(R), 1.0e-6))
                common = float(np.median(z))
                normalized = np.abs(z - common) / scale
                candidate = normalized <= gate
                if int(np.sum(candidate)) < self.config.tight_min_ranges or bool(np.all(candidate)):
                    break
                H = H[candidate]
                z = z[candidate]
                R = R[np.ix_(candidate, candidate)]
            if z.size < self.config.tight_min_ranges:
                return self.snapshot("tight", 0, math.nan)

        S = H @ self.P @ H.T + R
        try:
            Sinv_innov = np.linalg.solve(S, z)
            nis = float(z @ Sinv_innov)
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            return self.snapshot("tight", 0, math.nan)
        if nis > (self.config.measurement_gate_sigma ** 2 * z.size):
            return self.snapshot("tight", 0, math.sqrt(nis))
        dx = K @ z
        self._inject(dx[:15])
        self.clock_bias_m += dx[self.IDX_CLK]
        self.clock_drift_mps += dx[self.IDX_CLK_DRIFT]
        self.isb_m += dx[17:20]
        A = np.eye(self.state_dim) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2.0
        return self.snapshot("tight", int(z.size), math.sqrt(max(nis, 0.0)))


def _orthonormalize_local(C: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(C)
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def run_loose_coupling(
    imu: Iterable[IMURecord],
    gnss: Iterable[GNSSPVMeasurement],
    config: INSFilterConfig | None = None,
    init_window_s: float = 1.0,
) -> list[INSSnapshot]:
    imu_records = list(imu)
    gnss_records = list(gnss)
    if not imu_records:
        return []
    if not gnss_records:
        first = imu_records[0]
        initial = ECEFInertialState(first.time, np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
    else:
        first_gnss = gnss_records[0]
        window = [sample for sample in imu_records if first_gnss.time <= sample.time <= first_gnss.time + init_window_s]
        initial = initialize_state_from_gnss(first_gnss.time, first_gnss.position, first_gnss.velocity, window)
    filt = LooselyCoupledINS(initial, config)
    snapshots: list[INSSnapshot] = []
    gnss_index = 0
    for sample in imu_records:
        if sample.time < filt.state.time:
            continue
        snapshot = filt.propagate(sample)
        while gnss_index < len(gnss_records) and gnss_records[gnss_index].time <= sample.time + 0.5 * max(sample.dt, 0.0):
            measurement = gnss_records[gnss_index]
            if measurement.time >= filt.state.time - max(sample.dt, 0.0):
                snapshot = filt.update_pv(measurement)
            gnss_index += 1
        snapshots.append(snapshot)
    return snapshots
