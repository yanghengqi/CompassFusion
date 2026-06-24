import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set, List

from ..core.constants import CLIGHT


@dataclass(frozen=True)
class DsbRecord:
    sat: str          # like "C41"
    obs1: str         # like "C2I"
    obs2: str         # like "C6I"
    value_ns: float   # obs1-obs2 in ns


class BiasSinexDCB:
    """Minimal Bias-SINEX (BSX) reader for satellite DSB code biases.

    Supports lines like:
      DSB  Cxxx C41           C2I  C6I  2021:196:00000 ... ns  <value> <std>
    We only need sat+obs1+obs2+value (ns). Time validity is ignored for now (single-day file).
    """

    _re_dsb = re.compile(
        r"^\s*DSB\s+\S+\s+(?P<sat>[GRECJ]\d{2})\s+\S*\s*(?P<obs1>[CLD]\d[A-Z])\s+(?P<obs2>[CLD]\d[A-Z])\s+"
        r"(?P<beg>\d{4}:\d{3}:\d{5})\s+(?P<end>\d{4}:\d{3}:\d{5})\s+ns\s+(?P<val>[-+0-9.]+)"
    )

    def __init__(self, dsb: Dict[Tuple[str, str, str], float], sat_bias_m: Dict[Tuple[str, str], float]):
        self._dsb_ns = dsb  # (sat,obs1,obs2)->ns (obs1-obs2)
        self._bias_m = sat_bias_m  # (sat,obs)->bias (m) relative to chosen ref=0

    @staticmethod
    def from_file(path: str) -> "BiasSinexDCB":
        dsb: Dict[Tuple[str, str, str], float] = {}
        in_block = False
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("+BIAS/SOLUTION"):
                    in_block = True
                    continue
                if line.startswith("-BIAS/SOLUTION"):
                    in_block = False
                    continue
                if not in_block:
                    continue
                m = BiasSinexDCB._re_dsb.match(line)
                if not m:
                    continue
                sat = m.group("sat")
                obs1 = m.group("obs1")
                obs2 = m.group("obs2")
                val = float(m.group("val"))
                dsb[(sat, obs1, obs2)] = val
                dsb[(sat, obs2, obs1)] = -val

        # Build per-satellite relative biases (graph), choose reference heuristically.
        sat_bias_m: Dict[Tuple[str, str], float] = {}
        sats: Set[str] = {k[0] for k in dsb.keys()}
        for sat in sats:
            # adjacency: obs -> list of (nbr, dsb_ns) where dsb_ns = b_obs - b_nbr
            adj: Dict[str, List[Tuple[str, float]]] = {}
            obs_set: Set[str] = set()
            for (s, o1, o2), v in dsb.items():
                if s != sat:
                    continue
                adj.setdefault(o1, []).append((o2, v))
                obs_set.add(o1)
                obs_set.add(o2)

            if not obs_set:
                continue

            # choose reference
            ref = None
            if sat.startswith("C"):
                # prefer C2I then C1X then any
                for cand in ("C2I", "C1X", "C6I", "C7I", "C5X"):
                    if cand in obs_set:
                        ref = cand
                        break
            if ref is None:
                ref = sorted(obs_set)[0]

            # BFS to assign biases in ns relative to ref=0
            bias_ns: Dict[str, float] = {ref: 0.0}
            q = [ref]
            while q:
                u = q.pop(0)
                for v, dsb_uv in adj.get(u, []):
                    # dsb_uv = b_u - b_v  => b_v = b_u - dsb_uv
                    if v in bias_ns:
                        continue
                    bias_ns[v] = bias_ns[u] - dsb_uv
                    q.append(v)

            for o, bns in bias_ns.items():
                sat_bias_m[(sat, o)] = bns * 1e-9 * CLIGHT

        return BiasSinexDCB(dsb=dsb, sat_bias_m=sat_bias_m)

    def sat_code_bias_m(self, sat: str, obs: str) -> Optional[float]:
        return self._bias_m.get((sat, obs))

