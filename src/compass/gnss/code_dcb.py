"""CODE monthly P1-C1 / P2-C2 DCB reader (supplement when OSB is unavailable)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from ..core.constants import CLIGHT

_NS_TO_M = CLIGHT * 1e-9


@dataclass
class MonthlyCodeDcb:
    """Satellite differential code biases in metres (obs1 - obs2 convention)."""

    p1c1_m: Dict[str, float] = field(default_factory=dict)
    p2c2_m: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_files(cls, p1c1: Optional[str] = None, p2c2: Optional[str] = None) -> "MonthlyCodeDcb":
        obj = cls()
        if p1c1:
            obj.p1c1_m = cls._read_satellite_file(p1c1)
        if p2c2:
            obj.p2c2_m = cls._read_satellite_file(p2c2)
        return obj

    @classmethod
    def from_directory(cls, directory: str, year: int = 2021, month: int = 7) -> "MonthlyCodeDcb":
        yy = year % 100
        mm = f"{month:02d}"
        root = Path(directory)
        p1 = root / f"P1C1{yy}{mm}.DCB"
        p2 = root / f"P2C2{yy}{mm}_RINEX.DCB"
        if not p2.is_file():
            p2 = root / f"P2C2{yy}{mm}.DCB"
        return cls.from_files(str(p1) if p1.is_file() else None, str(p2) if p2.is_file() else None)

    @staticmethod
    def _read_satellite_file(path: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        re_sat = re.compile(r"^\s*([GRECJ]\d{2})\s+([-+0-9.]+)")
        with open(path, "r", encoding="ascii", errors="ignore") as stream:
            for line in stream:
                m = re_sat.match(line)
                if not m:
                    continue
                out[m.group(1).upper()] = float(m.group(2)) * _NS_TO_M
        return out

    def supplement_code_bias(self, system: str, sat_id: int, obs_code: str, osb_bias_m: float) -> float:
        """Return OSB bias, or a DCB-derived fallback when OSB is zero/missing."""
        if abs(osb_bias_m) > 1e-12:
            return osb_bias_m
        sat = f"{system}{sat_id:02d}"
        code = obs_code.upper()
        if code == "C1C" and sat in self.p1c1_m:
            return self.p1c1_m[sat]
        if code in ("C2C", "C2W", "C2X", "C2L") and sat in self.p2c2_m:
            return self.p2c2_m[sat]
        return 0.0
