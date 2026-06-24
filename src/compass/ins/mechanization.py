"""ECEF strapdown INS mechanization.

The routines in this module intentionally keep the navigation frame in ECEF.
That makes GNSS coupling straightforward because PPK/PPP/SPP positions and
velocities in this project are already ECEF quantities.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import math
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from ..core.constants import OMGE_GPS, wgs84
from ..core.transforms import ecef2llh

OMEGA_IE_E = np.array([0.0, 0.0, OMGE_GPS], dtype=float)
OMEGA_IE_MAT = np.array(
    [[0.0, -OMGE_GPS, 0.0], [OMGE_GPS, 0.0, 0.0], [0.0, 0.0, 0.0]],
    dtype=float,
)


@dataclass
class IMURecord:
    """One IMU sample.

    Gyro and accel are rates by default: rad/s and m/s^2. Use
    :meth:`as_delta` when code needs angular and velocity increments.
    """

    time: float
    gyro: np.ndarray
    accel: np.ndarray
    dt: float

    def as_delta(self) -> tuple[np.ndarray, np.ndarray]:
        return self.gyro * self.dt, self.accel * self.dt


@dataclass
class ECEFInertialState:
    """Nominal INS state in ECEF."""

    time: float
    position: np.ndarray
    velocity: np.ndarray
    C_be: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray

    def copy(self) -> "ECEFInertialState":
        return ECEFInertialState(
            float(self.time),
            self.position.copy(),
            self.velocity.copy(),
            self.C_be.copy(),
            self.gyro_bias.copy(),
            self.accel_bias.copy(),
        )


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _orthonormalize(C: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(C)
    Cn = u @ vt
    if np.linalg.det(Cn) < 0.0:
        u[:, -1] *= -1.0
        Cn = u @ vt
    return Cn


def rodrigues(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1.0e-12:
        return np.eye(3) + skew(rotation_vector)
    axis = rotation_vector / angle
    K = skew(axis)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def normal_gravity_ecef(position: np.ndarray) -> np.ndarray:
    """Normal gravity vector in ECEF, including the local vertical direction."""

    llh = ecef2llh(np.asarray(position, dtype=float))
    lat, lon, h = map(float, llh)
    sin_lat = math.sin(lat)
    g0 = 9.7803253359 * (1.0 + 0.00193185265241 * sin_lat * sin_lat) / math.sqrt(
        1.0 - wgs84.e2 * sin_lat * sin_lat
    )
    g = g0 * (1.0 - 2.0 * h / wgs84.a)
    down = -np.array(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=float,
    )
    return g * down


def ned_to_ecef_matrix(position: np.ndarray) -> np.ndarray:
    lat, lon, _ = map(float, ecef2llh(np.asarray(position, dtype=float)))
    east = np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=float)
    north = np.array([-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)], dtype=float)
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)], dtype=float)
    return np.column_stack((north, east, -up))


def attitude_from_ie_hpr_rfu(position: np.ndarray, heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build body-to-ECEF attitude from IE heading/pitch/roll, returning internal RFU axes."""

    heading = math.radians(float(heading_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))
    c_ned_frd = _rot_z(heading) @ _rot_y(pitch) @ _rot_x(-roll)
    c_frd_rfu = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)
    return _orthonormalize(ned_to_ecef_matrix(position) @ c_ned_frd @ c_frd_rfu)


def attitude_from_velocity_rfu(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Build a RFU body-to-ECEF attitude from GNSS velocity and local vertical."""

    body_up = np.array([0.0, 0.0, 1.0])
    body_forward = np.array([0.0, 1.0, 0.0])
    up_ecef = -normal_gravity_ecef(position)
    forward_ecef = np.asarray(velocity, dtype=float).reshape(3)
    return _triad(body_up, body_forward, up_ecef, forward_ecef)


def _axis_rotation_to_rfu(axis_order: str) -> np.ndarray:
    """Return a rotation that maps an input body axis order to internal RFU axes."""

    order = axis_order.lower()
    suffix = order[-3:] if len(order) >= 3 else order
    if suffix == "rfu":
        return np.eye(3)
    if suffix == "flu":
        return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    if suffix == "frd":
        return np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)
    if suffix == "rbd":
        return np.diag([1.0, -1.0, -1.0])
    if suffix == "lbu":
        return np.diag([-1.0, -1.0, 1.0])
    if suffix == "bru":
        return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    raise ValueError(f"Unsupported IMU axis order: {axis_order}")


def read_imu_file(
    path: str | Path,
    start_sow: float | None = None,
    end_sow: float | None = None,
    max_samples: int = 0,
    gyro_unit: str = "rad/s",
    axis_order: str = "rfu",
) -> list[IMURecord]:
    """Read a whitespace IMU file with columns Time Gyro_X Gyro_Y Gyro_Z Accel_X Accel_Y Accel_Z."""

    records: list[IMURecord] = []
    last_time: Optional[float] = None
    gyro_scale = {
        "rad/s": 1.0,
        "rps": 1.0,
        "dps": math.pi / 180.0,
        "deg/s": math.pi / 180.0,
    }.get(gyro_unit.lower())
    if gyro_scale is None:
        raise ValueError(f"Unsupported gyro unit: {gyro_unit}")
    axis_rotation = _axis_rotation_to_rfu(axis_order)
    with Path(path).open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                time = float(parts[0])
                values = np.asarray(parts[1:7], dtype=float)
            except ValueError:
                continue
            if start_sow is not None and time < start_sow:
                last_time = time
                continue
            if end_sow is not None and time > end_sow:
                break
            dt = 0.0 if last_time is None else time - last_time
            if dt <= 0.0 or dt > 1.0:
                dt = 0.0 if not records else records[-1].dt
            gyro = axis_rotation @ (values[:3].copy() * gyro_scale)
            accel = axis_rotation @ values[3:].copy()
            records.append(IMURecord(time, gyro, accel, dt))
            last_time = time
            if max_samples and len(records) >= max_samples:
                break
    if len(records) >= 2 and records[0].dt <= 0.0:
        records[0].dt = records[1].time - records[0].time
    return records


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-12:
        return np.asarray(fallback, dtype=float)
    return np.asarray(v, dtype=float) / n


def _triad(body_primary: np.ndarray, body_secondary: np.ndarray, ecef_primary: np.ndarray, ecef_secondary: np.ndarray) -> np.ndarray:
    b1 = _unit(body_primary, np.array([0.0, 0.0, 1.0]))
    b2 = body_secondary - b1 * float(np.dot(b1, body_secondary))
    b2 = _unit(b2, np.array([1.0, 0.0, 0.0]))
    b3 = np.cross(b1, b2)
    e1 = _unit(ecef_primary, np.array([0.0, 0.0, 1.0]))
    e2 = ecef_secondary - e1 * float(np.dot(e1, ecef_secondary))
    e2 = _unit(e2, np.array([1.0, 0.0, 0.0]))
    e3 = np.cross(e1, e2)
    return _orthonormalize(np.column_stack((e1, e2, e3)) @ np.column_stack((b1, b2, b3)).T)


def initialize_state_from_gnss(
    time: float,
    position: np.ndarray,
    velocity: np.ndarray,
    imu_window: Iterable[IMURecord] = (),
    gyro_bias: np.ndarray | None = None,
    accel_bias: np.ndarray | None = None,
) -> ECEFInertialState:
    """Initialize attitude from gravity and GNSS velocity in internal RFU body axes."""

    position = np.asarray(position, dtype=float).reshape(3)
    velocity = np.asarray(velocity, dtype=float).reshape(3)
    samples = list(imu_window)
    gyro_bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=float).reshape(3)
    accel_bias = np.zeros(3) if accel_bias is None else np.asarray(accel_bias, dtype=float).reshape(3)
    gravity = normal_gravity_ecef(position)
    if samples:
        mean_accel = np.mean([sample.accel for sample in samples], axis=0) - accel_bias
        mean_gyro = np.mean([sample.gyro for sample in samples], axis=0)
        if np.linalg.norm(gyro_bias) == 0.0 and np.linalg.norm(velocity) < 0.2:
            gyro_bias = mean_gyro.copy()
    else:
        mean_accel = np.array([0.0, 0.0, 9.80665])
    speed = float(np.linalg.norm(velocity))
    body_forward = np.array([0.0, 1.0, 0.0])
    if speed > 0.2:
        C_be = _triad(mean_accel, body_forward, -gravity, velocity)
    else:
        llh = ecef2llh(position)
        lon = float(llh[1])
        east = np.array([-math.sin(lon), math.cos(lon), 0.0])
        C_be = _triad(mean_accel, body_forward, -gravity, east)
    return ECEFInertialState(float(time), position, velocity, C_be, gyro_bias, accel_bias)


def strapdown_step_ecef(state: ECEFInertialState, sample: IMURecord) -> ECEFInertialState:
    """Propagate one ECEF strapdown step with a midpoint-style velocity update."""

    dt = float(sample.dt)
    if dt <= 0.0:
        return state.copy()
    omega_b = np.asarray(sample.gyro, dtype=float) - state.gyro_bias
    force_b = np.asarray(sample.accel, dtype=float) - state.accel_bias
    C_old = state.C_be
    C_mid = _orthonormalize(rodrigues(-OMEGA_IE_E * (0.5 * dt)) @ C_old @ rodrigues(omega_b * (0.5 * dt)))
    specific_force_e = C_mid @ force_b
    coriolis = -2.0 * np.cross(OMEGA_IE_E, state.velocity)
    acceleration = specific_force_e + normal_gravity_ecef(state.position) + coriolis
    velocity = state.velocity + acceleration * dt
    position = state.position + 0.5 * (state.velocity + velocity) * dt
    C_new = _orthonormalize(rodrigues(-OMEGA_IE_E * dt) @ C_old @ rodrigues(omega_b * dt))
    return ECEFInertialState(sample.time, position, velocity, C_new, state.gyro_bias.copy(), state.accel_bias.copy())


def iter_mechanization(initial: ECEFInertialState, imu: Iterable[IMURecord]) -> Iterator[ECEFInertialState]:
    state = initial.copy()
    for sample in imu:
        state = strapdown_step_ecef(state, sample)
        yield state
