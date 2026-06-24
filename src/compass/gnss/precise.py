"""Readers and interpolator for SP3 precise orbit and RINEX clock products."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

_GPS_EPOCH = datetime(1980, 1, 6)


def _gps_seconds(dt: datetime) -> float:
    return (dt - _GPS_EPOCH).total_seconds()


def _epoch(line: str) -> float:
    p = line[1:].split()
    sec = float(p[5])
    dt = datetime(int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]), int(sec), int(round((sec % 1) * 1e6)))
    return _gps_seconds(dt)


@dataclass
class PreciseProducts:
    """Precise satellite states keyed by RINEX satellite id (for example G01)."""

    orbits: Dict[str, list[Tuple[float, np.ndarray, float]]] = field(default_factory=dict)
    clocks: Dict[str, list[Tuple[float, float]]] = field(default_factory=dict)
    max_orbit_gap: float = 1200.0
    max_clock_gap: float = 120.0
    clock_edge_tolerance: float = 1.0

    @classmethod
    def from_files(cls, sp3_files: Iterable[str], clk_files: Iterable[str] = ()) -> "PreciseProducts":
        obj = cls()
        for path in sp3_files:
            obj.read_sp3(path)
        for path in clk_files:
            obj.read_clk(path)
        obj._sort()
        return obj

    def _sort(self) -> None:
        for values in self.orbits.values():
            values.sort(key=lambda x: x[0])
        for values in self.clocks.values():
            values.sort(key=lambda x: x[0])

    def read_sp3(self, path: str) -> None:
        time = None
        with open(path, "r", encoding="ascii", errors="replace") as stream:
            for line in stream:
                if line.startswith("*"):
                    time = _epoch(line)
                elif time is not None and line.startswith("P") and len(line) >= 46:
                    sat = line[1:4].strip().upper()
                    try:
                        xyz = np.array([float(line[4:18]), float(line[18:32]), float(line[32:46])]) * 1000.0
                        clk_us = float(line[46:60]) if len(line) >= 60 else 999999.999999
                    except ValueError:
                        continue
                    if np.all(np.isfinite(xyz)) and np.max(np.abs(xyz)) < 1e8:
                        clk = np.nan if abs(clk_us) >= 999999.0 else clk_us * 1e-6
                        self.orbits.setdefault(sat, []).append((time, xyz, clk))
        self._sort()

    def read_clk(self, path: str) -> None:
        with open(path, "r", encoding="ascii", errors="replace") as stream:
            in_header = True
            for line in stream:
                if in_header:
                    in_header = "END OF HEADER" not in line
                    continue
                if not line.startswith("AS "):
                    continue
                p = line.split()
                if len(p) < 10:
                    continue
                try:
                    sec = float(p[7])
                    dt = datetime(int(p[2]), int(p[3]), int(p[4]), int(p[5]), int(p[6]), int(sec), int(round((sec % 1) * 1e6)))
                    self.clocks.setdefault(p[1].upper(), []).append((_gps_seconds(dt), float(p[9])))
                except (ValueError, OverflowError):
                    continue
        self._sort()

    @staticmethod
    def _linear(rows, time: float, value_index: int, max_gap: float, edge_tolerance: float = 0.0):
        ts = np.asarray([r[0] for r in rows])
        if len(ts) == 0 or time < ts[0] - edge_tolerance or time > ts[-1] + edge_tolerance:
            return None
        if time <= ts[0]:
            return rows[0][value_index]
        if time >= ts[-1]:
            return rows[-1][value_index]
        j = int(np.searchsorted(ts, time))
        a, b = rows[j - 1], rows[j]
        if b[0] - a[0] > max_gap:
            return None
        f = (time - a[0]) / (b[0] - a[0])
        return (1.0 - f) * a[value_index] + f * b[value_index]

    def state_full(self, system: str, sat_id: int, time: float) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        """Interpolate COM position/velocity and relativistically corrected clock."""
        sat = f"{system}{sat_id:02d}"
        rows = self.orbits.get(sat)
        if not rows:
            return None
        ts = np.asarray([r[0] for r in rows])
        j = int(np.searchsorted(ts, time))
        lo, hi = max(0, j - 5), min(len(rows), j + 5)
        sample = rows[lo:hi]
        if len(sample) < 2 or min(abs(time - r[0]) for r in sample) > self.max_orbit_gap:
            return None
        order = min(8, len(sample))
        sample = sorted(sample, key=lambda r: abs(r[0] - time))[:order]
        tx = np.asarray([(r[0] - time) / 900.0 for r in sample])
        xyz = np.vstack([r[1] for r in sample])
        degree = min(5, len(sample) - 1)
        coeff = [np.polynomial.polynomial.polyfit(tx, xyz[:, k], degree) for k in range(3)]
        pos = np.asarray([c[0] for c in coeff])
        vel = np.asarray([c[1] / 900.0 if len(c) > 1 else 0.0 for c in coeff])
        clk_rows = self.clocks.get(sat)
        source = clk_rows
        clk = self._linear(
            clk_rows, time, 1, self.max_clock_gap, self.clock_edge_tolerance
        ) if clk_rows else None
        if clk is None:
            source = [(r[0], r[2]) for r in rows if np.isfinite(r[2])]
            clk = self._linear(
                source, time, 1, self.max_orbit_gap, self.clock_edge_tolerance
            ) if source else None
        if clk is None or not np.isfinite(clk):
            return None
        clk = float(clk) - 2.0 * float(np.dot(pos, vel)) / (299792458.0 ** 2)
        st = np.asarray([r[0] for r in source])
        jj = int(np.searchsorted(st, time))
        jj = min(max(jj, 1), len(source) - 1)
        clk_drift = ((source[jj][1] - source[jj - 1][1]) /
                     (source[jj][0] - source[jj - 1][0])) if len(source) > 1 else 0.0
        return pos, vel, clk, float(clk_drift)
    def state(self, system: str, sat_id: int, time: float) -> Optional[Tuple[np.ndarray, float]]:
        full = self.state_full(system, sat_id, time)
        return None if full is None else (full[0], full[2])