#!/usr/bin/env python3
"""Diagnose POTS PPP failures without modifying ppp/spp core."""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compass.gnss.bias_osb import OsbCalibration
from compass.gnss.code_dcb import MonthlyCodeDcb
from compass.gnss.orbex_attitude import OrbexAttitude
from compass.gnss.ppp import PPPConfig, PPPKalmanFilter
from compass.gnss.ppp_models import AntexCalibration, EarthRotationParameters
from compass.gnss.precise import PreciseProducts
from compass.gnss.spp import SPPSolver
from compass.io.rinex_native import RINEXNativeReader

SNX = np.array([3800689.375, 882077.650, 5028791.485])


def load_stack():
    r = RINEXNativeReader()
    nav = r.read_nav(ROOT / "data/nav/brdm1960.21p")
    products = PreciseProducts.from_files(
        [ROOT / "data/sp3/COD0MGXFIN_20211960000_01D_05M_ORB.SP3"],
        [ROOT / "data/clk/COD0MGXFIN_20211960000_01D_30S_CLK.CLK"],
    )
    flt = PPPKalmanFilter(
        products,
        nav,
        PPPConfig(),
        AntexCalibration.from_file(ROOT / "data/atx/igs20.atx"),
        EarthRotationParameters.from_file(ROOT / "data/erp/COD0MGXFIN_20211960000_03D_12H_ERP.ERP"),
        "JAVRINGANT_G5T",
        osb=OsbCalibration.from_file(ROOT / "data/bia/COD0MGXFIN_20211960000_01D_01D_OSB.BIA"),
        obx=OrbexAttitude.from_file(ROOT / "data/bia/COD0MGXFIN_20211960000_01D_15M_ATT.OBX"),
        dcb=MonthlyCodeDcb.from_directory(ROOT / "data/dcb", 2021, 10),
    )
    return r, nav, flt


def filter_epoch(epoch, systems: str):
    e = copy.copy(epoch)
    e.observations = [o for o in epoch.observations if o.system in set(systems)]
    return e


def check_if_mismatch(flt: PPPKalmanFilter, epoch):
    """Compare _combinations IF freqs vs _frequency_pair in _build second loop."""
    bad = []
    for ob in epoch.observations:
        comb = flt._combinations(ob)
        if comb is None:
            continue
        f1c, f2c, atx1, atx2 = comb[5], comb[6], comb[7], comb[8]
        pair = flt._frequency_pair(ob)
        if pair is None:
            bad.append((ob.system, ob.sat_id, "pair_none", f1c, f2c, atx1, atx2))
            continue
        f1p, f2p, ap1, ap2 = pair
        if abs(f1c - f1p) > 1e3 or abs(f2c - f2p) > 1e3 or atx1 != ap1 or atx2 != ap2:
            raw = sorted((ob.raw_observations or {}).keys())
            bad.append((ob.system, ob.sat_id, raw[:8], f"comb={f1c/1e6:.3f}/{f2c/1e6:.3f}/{atx1}/{atx2}",
                        f"pair={f1p/1e6:.3f}/{f2p/1e6:.3f}/{ap1}/{ap2}"))
    return bad


def epoch_diag(flt, epoch, approx):
    t = epoch.week * 604800.0 + epoch.timestamp
    if flt.x is None:
        flt._initialize(epoch, approx)
    spp = SPPSolver(elev_mask=flt.config.elevation_mask_deg)
    t0 = time.perf_counter()
    spp_pos, spp_clk, spp_ok = spp.solve(epoch, flt.nav, flt.x[:3])
    spp_ms = (time.perf_counter() - t0) * 1000
    cand, H, v, var, kinds, keys = flt._build(epoch, t)
    prefit = postfit = used = 0
    if len(v):
        sdiag = np.einsum("ij,jk,ik->i", H, flt.P, H) + var
        norms = np.abs(v) / np.sqrt(np.maximum(sdiag, 1e-12))
        prefit = int(np.sum(norms <= flt.config.prefit_gate_sigma))
        mask, post = flt._robust_update(H, v, var, kinds, keys)
        used = len(set(k for i, k in enumerate(keys) if mask[i]))
        if mask.any():
            score = np.abs(post[mask]) / np.sqrt(var[mask])
            postfit = int(np.sum(score <= flt.config.postfit_code_gate_sigma))
    pos_err = np.linalg.norm(flt.x[:3] - SNX)
    spp_err = np.linalg.norm(spp_pos - SNX) if spp_ok else np.nan
    return {
        "sow": epoch.timestamp,
        "pos": flt.x[:3].copy(),
        "pos_err_snx": pos_err,
        "spp_err_snx": spp_err,
        "spp_ms": spp_ms,
        "cand": len(cand),
        "prefit_ok": prefit,
        "nmeas": len(v),
        "used_sats": used,
        "clk": flt.x[flt.IDX_CLK],
    }


def main():
    reader, nav, flt = load_stack()
    epochs = reader.read_obs(ROOT / "data/obs/pots1960.21o", max_epochs=20)
    ep0 = filter_epoch(epochs[0], "GEC")

    print("=== 1. IF 频点: _combinations vs _build._frequency_pair ===")
    mism = check_if_mismatch(flt, ep0)
    print(f"  不匹配卫星数: {len(mism)} / {len(ep0.observations)}")
    for row in mism[:8]:
        print(" ", row)

    print("\n=== 2. 首历元 SPP / 滤波诊断 ===")
    d = epoch_diag(flt, ep0, SNX)
    for k, v in d.items():
        print(f"  {k}: {v}")

    print("\n=== 3. 连续 process() 20 历元（原版逻辑）===")
    flt2, _, _ = load_stack()
    flt2 = load_stack()[2]
    spp_times = []
    for i, ep in enumerate(epochs[:20], 1):
        epf = filter_epoch(ep, "GEC")
        t0 = time.perf_counter()
        sol = flt2.process(epf, SNX)
        dt = (time.perf_counter() - t0) * 1000
        spp_times.append(dt)
        if i <= 8 or sol.status:
            print(
                f"  {i:2d} sow={ep.timestamp:.0f} status={sol.status} ns={sol.satellites} "
                f"pos_err={np.linalg.norm(sol.position-SNX):.1f}m code={sol.code_rms_m:.1f}m "
                f"time={dt:.0f}ms fail={flt2.consecutive_failures}"
            )
    print(f"  平均耗时 {np.mean(spp_times):.0f} ms/历元, 最大 {np.max(spp_times):.0f} ms")

    print("\n=== 4. 仅 GPS+Gal (GE) 首历元 ===")
    flt3 = load_stack()[2]
    ep_ge = filter_epoch(epochs[0], "GE")
    d = epoch_diag(flt3, ep_ge, SNX)
    print(f"  cand={d['cand']} used_sats={d['used_sats']} pos_err={d['pos_err_snx']:.2f}m")


if __name__ == "__main__":
    main()
