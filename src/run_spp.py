#!/usr/bin/env python3
"""SPP 批处理：读 RINEX 观测 + 导航，逐历元输出 ECEF/LLH。"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np

from compass.core.transforms import ecef2llh, llh2ecef
from compass.gnss.spp import SPPSolver
from compass.io.rinex_native import RINEXNativeReader

# RINEX 系统字：G GPS, R GLONASS, E Galileo, C BDS, J QZSS, I IRNSS
_SYSTEM_LETTERS = frozenset("GRECIJ")


def _parse_approx(s: str) -> np.ndarray:
    """ECEF 米: 'x,y,z'"""
    parts = [float(x) for x in s.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("需要三个数: x,y,z (米, ECEF)")
    return np.array(parts, dtype=float)


def _parse_approx_llh(s: str) -> np.ndarray:
    """度/米: 'lat,lon,h'"""
    parts = [float(x) for x in s.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("需要三个数: lat,lon,h (deg,deg,m)")
    lat, lon, h = parts
    llh = np.array([np.radians(lat), np.radians(lon), h], dtype=float)
    return llh2ecef(llh)


def _parse_systems(s: str) -> frozenset[str]:
    """连写或多系统组合，如 GRE、GRCE、G；逗号/空格仅作分隔，按字母集合去重。"""
    if not s or not str(s).strip():
        raise argparse.ArgumentTypeError("不能为空")
    letters = [c.upper() for c in s if c.isalpha()]
    if not letters:
        raise argparse.ArgumentTypeError("至少需要一个大写字母 G/R/E/C/J/I")
    bad = [c for c in letters if c not in _SYSTEM_LETTERS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"非法系统字母: {sorted(set(bad))}，允许 G R E C J I"
        )
    return frozenset(letters)


def _filter_systems(r_obs, systems: frozenset[str] | None):
    if systems is None:
        return r_obs
    kept = [o for o in r_obs.observations if o.system in systems]
    return replace(r_obs, observations=kept)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="COMPASS-Python SPP（独立包）")
    p.add_argument("--obs", required=True, help="RINEX 观测文件 (.o)")
    p.add_argument("--nav", required=True, help="RINEX 导航文件 (.p/.n 等)")
    p.add_argument(
        "--out",
        default="",
        help="输出 CSV 路径；默认 stdout（列: week,sow,x,y,z,lat_deg,lon_deg,h_m,status）",
    )
    p.add_argument("--elev", type=float, default=15.0, help="高度角截止角 (度)")
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--approx",
        type=_parse_approx,
        default=None,
        metavar="X,Y,Z",
        help="概略位置 ECEF (米)，逗号或空格分隔",
    )
    g.add_argument(
        "--approx-llh",
        type=_parse_approx_llh,
        dest="approx_llh",
        default=None,
        metavar="LAT,LON,H",
        help="概略位置 纬度、经度(度)、高程(米)",
    )
    p.add_argument("--max-epochs", type=int, default=0, help="最多解算历元数，0 表示全部")
    p.add_argument(
        "--systems",
        type=_parse_systems,
        default=None,
        metavar="GRE",
        help=(
            "仅使用指定系统，连写即可，任意组合（如 G、GE、GRE、GRC、GRCE）；"
            "G/R/E/C/J/I 对应 GPS/GLONASS/Galileo/BDS/QZSS/IRNSS；"
            "省略则使用观测文件中的全部系统"
        ),
    )
    args = p.parse_args(argv)

    if not os.path.isfile(args.obs):
        print(f"找不到观测文件: {args.obs}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.nav):
        print(f"找不到导航文件: {args.nav}", file=sys.stderr)
        return 2

    approx = args.approx
    if args.approx_llh is not None:
        approx = args.approx_llh

    reader = RINEXNativeReader()
    nav_list = reader.read_nav(args.nav)
    obs_list = reader.read_obs(args.obs)

    solver = SPPSolver(
        elev_mask=args.elev,
        ion_gps=reader.nav_ion_gps,
        ion_bds=reader.nav_ion_bds,
    )

    fout = open(args.out, "w", encoding="utf-8", newline="\n") if args.out else sys.stdout
    try:
        fout.write("week,sow,x_m,y_m,z_m,lat_deg,lon_deg,h_m,status\n")
        n = 0
        for r_obs in obs_list:
            ro = _filter_systems(r_obs, args.systems)
            pos, _clk, stat = solver.solve(ro, nav_list, approx)
            llh = ecef2llh(pos) if stat == 1 else (np.nan, np.nan, np.nan)
            lat_d = float(np.degrees(llh[0])) if stat == 1 else float("nan")
            lon_d = float(np.degrees(llh[1])) if stat == 1 else float("nan")
            h_m = float(llh[2]) if stat == 1 else float("nan")
            fout.write(
                f"{r_obs.week},{r_obs.timestamp:.3f},"
                f"{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f},"
                f"{lat_d:.8f},{lon_d:.8f},{h_m:.4f},{stat}\n"
            )
            n += 1
            if args.max_epochs and n >= args.max_epochs:
                break
    finally:
        if fout is not sys.stdout:
            fout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
