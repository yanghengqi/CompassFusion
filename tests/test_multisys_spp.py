#!/usr/bin/env python3
"""
多系统 SPP 组合测试：同一份 RINEX 上跑多种系统筛选，统计成功率；
可选与同历元 ECEF 参考 .pos 对齐，统计 3D 位置差。

默认：观测/导航优先 ../compass-python/test/，否则本包 data/；
参考 .pos 默认本包 data/KVH0_fix.pos（若不存在再尝试 compass-python/test 下 .pos）。

用法:
  python test_multisys_spp.py
  python test_multisys_spp.py --truth path/to/ref.pos --epochs 100
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np

from compass.core.transforms import ecef2llh
from compass.gnss.spp import SPPSolver
from compass.io.rinex_native import RINEXNativeReader


# 多系统组合（≥2 个字母或 ALL）；可按需增删
MULTISYS_COMBOS: list[tuple[str, frozenset[str] | None]] = [
    ("GE", frozenset("GE")),
    ("GR", frozenset("GR")),
    ("GC", frozenset("GC")),
    ("EC", frozenset("EC")),
    ("RC", frozenset("RC")),
    ("GRE", frozenset("GRE")),
    ("GRC", frozenset("GRC")),
    ("GCE", frozenset("GCE")),
    ("RCE", frozenset("RCE")),
    ("GRCE", frozenset("GRCE")),
    ("ALL", None),
]


def _default_paths(here: str) -> tuple[str, str, str]:
    test_dir = os.path.normpath(os.path.join(here, "..", "compass-python", "test"))
    t_truth_alt = os.path.join(test_dir, "KVH0_G_DF_BRD_FIX.pos")
    t_obs = os.path.join(test_dir, "KVH01960.21o")
    t_nav = os.path.join(test_dir, "brdm1960.21p")
    if os.path.isfile(t_obs) and os.path.isfile(t_nav):
        obs, nav = t_obs, t_nav
    else:
        obs = os.path.join(here, "data", "KVH01960.21o")
        nav = os.path.join(here, "data", "brdm1960.21p")
    ref_local = os.path.join(here, "data", "KVH0_fix.pos")
    if os.path.isfile(ref_local):
        truth = ref_local
    elif os.path.isfile(t_truth_alt):
        truth = t_truth_alt
    else:
        truth = ""
    return obs, nav, truth


def _filter(r_obs, systems: frozenset[str] | None):
    if systems is None:
        return r_obs
    kept = [o for o in r_obs.observations if o.system in systems]
    return replace(r_obs, observations=kept)


def _count_by_sys(obs_list, systems: frozenset[str] | None) -> dict[str, int]:
    r0 = obs_list[0]
    ro = _filter(r0, systems)
    cnt: dict[str, int] = {}
    for o in ro.observations:
        cnt[o.system] = cnt.get(o.system, 0) + 1
    return cnt


def load_ref_pos(path: str, fix_only: bool = True) -> dict[tuple[int, int], np.ndarray]:
    """RTKLIB 风格 .pos：GPST 周、秒内秒、ECEF；键 (week, int(round(sod)))。

    fix_only=True 时仅保留 Q==1 行。
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("%") or not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                week = int(parts[0].strip())
                sod = float(parts[1].strip())
                x = float(parts[2].strip())
                y = float(parts[3].strip())
                z = float(parts[4].strip())
                q = int(parts[5].strip())
            except ValueError:
                continue
            if fix_only and q != 1:
                continue
            key = (week, int(round(sod)))
            out[key] = np.array([x, y, z], dtype=float)
    return out


def _first_obs_index_with_ref(
    obs_list, ref: dict[tuple[int, int], np.ndarray]
) -> int:
    for i, r in enumerate(obs_list):
        k = (r.week, int(round(r.timestamp)))
        if k in ref:
            return i
    return -1


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    here = os.path.dirname(os.path.abspath(__file__))
    def_obs, def_nav, def_truth = _default_paths(here)

    p = argparse.ArgumentParser(description="多系统 SPP 组合测试（可选参考 .pos 对比）")
    p.add_argument("--obs", default=def_obs)
    p.add_argument("--nav", default=def_nav)
    p.add_argument(
        "--truth",
        default=def_truth,
        metavar="PATH",
        help="参考 ECEF .pos（默认 data/KVH0_fix.pos 或 compass-python/test 下同名类文件）",
    )
    p.add_argument(
        "--truth-all-q",
        action="store_true",
        help="不限制 Q==1，所有有效数据行均参与对比",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="从首个与参考时间对齐的历元起算的历元数；≤0 表示直到观测文件末尾（完整对比）",
    )
    p.add_argument("--elev", type=float, default=15.0, help="高度角截止角 (度)")
    args = p.parse_args(argv)

    if not os.path.isfile(args.obs):
        print(f"缺少观测文件: {args.obs}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.nav):
        print(f"缺少导航文件: {args.nav}", file=sys.stderr)
        return 2

    truth: dict[tuple[int, int], np.ndarray] | None = None
    if args.truth:
        if not os.path.isfile(args.truth):
            print(f"未找到参考 .pos，跳过位置对比: {args.truth}", file=sys.stderr)
        else:
            truth = load_ref_pos(args.truth, fix_only=not args.truth_all_q)

    reader = RINEXNativeReader()
    nav_list = reader.read_nav(args.nav)
    obs_list = reader.read_obs(args.obs)
    if not obs_list:
        print("观测历元为空", file=sys.stderr)
        return 2

    i0 = 0
    rest = len(obs_list)
    if args.epochs <= 0:
        n_ep = rest
    else:
        n_ep = min(args.epochs, rest)
    if truth:
        i0 = _first_obs_index_with_ref(obs_list, truth)
        if i0 < 0:
            print("观测与参考 .pos 时间无交集，跳过位置对比", file=sys.stderr)
            truth = None
            i0 = 0
            n_ep = min(args.epochs, len(obs_list)) if args.epochs > 0 else len(obs_list)
        else:
            rest = len(obs_list) - i0
            n_ep = rest if args.epochs <= 0 else min(args.epochs, rest)

    solver = SPPSolver(
        elev_mask=args.elev,
        ion_gps=reader.nav_ion_gps,
        ion_bds=reader.nav_ion_bds,
    )

    print(
        f"数据: obs={args.obs}\n"
        f"      nav={args.nav}\n"
        f"参考: {args.truth or '(未指定)'}\n"
        f"历元: 从索引 {i0} 起连续 {n_ep} 个  elev_mask={args.elev}°\n"
        f"solve 未传 approx_pos，使用 RINEX 头 + 内核逻辑。\n"
    )

    hdr = (
        f"{'组合':<8} {'星数(首段首历元)':<18} {'ep0':>5} "
        f"{'成功':>6}/{n_ep:<4} {'lat°(ep0)':>12} {'lon°(ep0)':>12} {'h_m(ep0)':>10}"
    )
    if truth:
        hdr += f" {'n_3d':>6} {'mean_m':>8} {'rms_m':>8}"
    print(hdr)
    print("-" * (98 if truth else 80))
    if truth:
        n_ref_epoch = sum(
            1
            for k in range(n_ep)
            if (
                obs_list[i0 + k].week,
                int(round(obs_list[i0 + k].timestamp)),
            )
            in truth
        )
        print(
            f"注: 本段 {n_ep} 个观测历元中，有 {n_ref_epoch} 个在参考 .pos 中存在同一时刻；"
            f"n_3d 为「SPP 成功且该时刻有参考」的个数，故 n_3d ≤ min(成功,{n_ref_epoch})。\n"
        )

    for label, systems in MULTISYS_COMBOS:
        ro0 = _filter(obs_list[i0], systems)
        n_sat = len(ro0.observations)
        sys_str = ",".join(
            f"{k}:{v}" for k, v in sorted(_count_by_sys([obs_list[i0]], systems).items())
        )

        ok = 0
        st0 = -1
        lat0 = lon0 = h0 = float("nan")
        errs: list[float] = []

        for k in range(n_ep):
            i = i0 + k
            ro = _filter(obs_list[i], systems)
            pos, _clk, st = solver.solve(ro, nav_list, None)
            if st == 1:
                ok += 1
            if k == 0:
                st0 = st
                if st == 1:
                    a, b, c = ecef2llh(pos)
                    lat0, lon0, h0 = float(np.degrees(a)), float(np.degrees(b)), float(c)

            if truth and st == 1 and np.linalg.norm(pos) > 1e3:
                tk = (obs_list[i].week, int(round(obs_list[i].timestamp)))
                if tk in truth:
                    errs.append(float(np.linalg.norm(pos - truth[tk])))

        sat_col = f"{n_sat} ({sys_str})" if sys_str else str(n_sat)
        row = (
            f"{label:<8} {sat_col:<18} {st0:>5} {ok:>6}/{n_ep:<4} "
            f"{lat0:>12.6f} {lon0:>12.6f} {h0:>10.2f}"
        )
        if truth:
            if errs:
                a = np.array(errs, dtype=float)
                row += f" {len(errs):>6} {a.mean():>8.2f} {float(np.sqrt((a * a).mean())):>8.2f}"
            else:
                row += f" {'0':>6} {'—':>8} {'—':>8}"
        print(row)

    print("-" * (98 if truth else 80))
    print(
        "ep0: 该段首历元 status（1=成功）；成功列: 该段内 status=1 个数。\n"
        "n_3d/mean/rms: 仅统计与参考 .pos 按周+整秒对齐的历元。\n"
        "ALL 含 QZSS(J) 等时可能全失败，可改用 GRCE。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
