"""CODE/IGS OSB (Observable-Specific Bias) reader for Bias-SINEX .BIA files."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from ..core.constants import CLIGHT

_NS_TO_M = CLIGHT * 1e-9


@dataclass
class OsbCalibration:
    """Satellite code/phase OSB lookup keyed by (PRN, RINEX obs code)."""

    code_bias_m: Dict[Tuple[str, str], float] = field(default_factory=dict)
    phase_bias_m: Dict[Tuple[str, str], float] = field(default_factory=dict)

    _re_osb = re.compile(
        r"^\s*OSB\s+\S+\s+(?P<sat>[GRECJ]\d{2})\s+\S*\s+(?P<obs>[CL]\w{2})\s+"
        r"\d{4}:\d{3}:\d{5}\s+\d{4}:\d{3}:\d{5}\s+ns\s+(?P<val>[-+0-9.]+)"
    )

    @classmethod
    def from_file(cls, path: str) -> "OsbCalibration":
        obj = cls()
        in_block = False
        with open(path, "r", encoding="ascii", errors="ignore") as stream:
            for line in stream:
                if line.startswith("+BIAS/SOLUTION"):
                    in_block = True
                    continue
                if line.startswith("-BIAS/SOLUTION"):
                    break
                if not in_block:
                    continue
                m = cls._re_osb.match(line)
                if not m:
                    continue
                sat = m.group("sat")
                obs = m.group("obs").upper()
                val_m = float(m.group("val")) * _NS_TO_M
                key = (sat, obs)
                if obs[0] == "C":
                    obj.code_bias_m[key] = val_m
                elif obs[0] == "L":
                    obj.phase_bias_m[key] = val_m
        if not obj.code_bias_m and not obj.phase_bias_m:
            raise ValueError(f"No OSB records in {path}")
        return obj

    @staticmethod
    def obs_code(rinex_kind: str, rinex_suffix: str) -> str:
        return f"{rinex_kind[0].upper()}{rinex_suffix.upper()}"

    def code_bias(self, system: str, sat_id: int, obs_code: str) -> float:
        return self.code_bias_m.get((f"{system}{sat_id:02d}", obs_code.upper()), 0.0)

    def phase_bias(self, system: str, sat_id: int, obs_code: str) -> float:
        return self.phase_bias_m.get((f"{system}{sat_id:02d}", obs_code.upper()), 0.0)
