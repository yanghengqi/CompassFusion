"""ORBEX satellite attitude (quaternion) reader for CODE MGEX OBX products."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

GPS_EPOCH = datetime(1980, 1, 6)
LEAP_SECONDS = 18.0


def _gps_seconds(dt: datetime) -> float:
    return (dt - GPS_EPOCH).total_seconds() + LEAP_SECONDS


def _parse_epoch_line(line: str) -> Optional[float]:
    if not line.startswith("##"):
        return None
    p = line.split()
    if len(p) < 7:
        return None
    try:
        sec = float(p[6])
        dt = datetime(int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(sec), int(round((sec % 1) * 1e6)))
        return _gps_seconds(dt)
    except (ValueError, TypeError):
        return None


def body_axes_from_quaternion(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Body x/y/z unit vectors in ECEF from scalar-first quaternion (body -> ECEF)."""
    q0, q1, q2, q3 = [float(v) for v in q]
    r = np.array([
        [1 - 2 * (q2 * q2 + q3 * q3), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 * q1 + q2 * q2)],
    ], float)
    return r[:, 0], r[:, 1], r[:, 2]


@dataclass
class OrbexAttitude:
    epochs: List[float] = field(default_factory=list)
    attitudes: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str) -> "OrbexAttitude":
        obj = cls()
        epoch_idx = -1
        with open(path, "r", encoding="ascii", errors="ignore") as stream:
            for line in stream:
                t = _parse_epoch_line(line)
                if t is not None:
                    obj.epochs.append(t)
                    epoch_idx += 1
                    continue
                if not line.startswith("ATT "):
                    continue
                p = line.split()
                if len(p) < 7:
                    continue
                sat = p[1].strip().upper()
                try:
                    q = np.asarray([float(p[3]), float(p[4]), float(p[5]), float(p[6])], float)
                except ValueError:
                    continue
                n = float(np.linalg.norm(q))
                if n <= 0.0:
                    continue
                q /= n
                obj.attitudes.setdefault(sat, []).append(q)
        if not obj.epochs:
            raise ValueError(f"No attitude epochs in {path}")
        for sat, rows in obj.attitudes.items():
            if len(rows) != len(obj.epochs):
                raise ValueError(f"ORBEX attitude count mismatch for {sat}")
        return obj

    def body_axes(self, system: str, sat_id: int, gps_seconds: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        sat = f"{system}{sat_id:02d}"
        series = self.attitudes.get(sat)
        if not series:
            return None
        times = self.epochs
        if gps_seconds <= times[0]:
            return body_axes_from_quaternion(series[0])
        if gps_seconds >= times[-1]:
            return body_axes_from_quaternion(series[-1])
        j = int(np.searchsorted(times, gps_seconds))
        t0, t1 = times[j - 1], times[j]
        f = (gps_seconds - t0) / (t1 - t0) if t1 > t0 else 0.0
        q = (1.0 - f) * series[j - 1] + f * series[j]
        n = float(np.linalg.norm(q))
        if n <= 0.0:
            return None
        return body_axes_from_quaternion(q / n)
