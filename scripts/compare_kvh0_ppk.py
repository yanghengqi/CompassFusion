#!/usr/bin/env python3
"""Compare Python PPK against the overlapping KVH0 GNSS fixed solution."""
from __future__ import annotations
import csv,math,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def reference(path):
    result={}
    for line in path.open(errors="ignore"):
        parts=[item.strip() for item in line.split(",")]
        if len(parts)<6:continue
        try:result[float(parts[1])]=(np.asarray(parts[2:5],float),int(parts[5]))
        except ValueError:continue
    return result

def main():
    result_path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"data"/"ppk_result_KVH0.csv"
    truth=reference(ROOT/"data"/"KVH0_fix.pos");groups={"all":[],"float":[],"fix":[]}
    with result_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            sow=float(row["sow"])
            if row["status"]!="1" or sow not in truth:continue
            error=np.asarray([row["x_m"],row["y_m"],row["z_m"]],float)-truth[sow][0]
            groups["all"].append(error);groups["fix" if row["fix_status"]=="1" else "float"].append(error)
    for name,errors in groups.items():
        if not errors:continue
        values=np.asarray(errors);distance=np.linalg.norm(values,axis=1)
        print(f"{name}: n={len(values)} 3D_RMS={math.sqrt(np.mean(distance**2)):.3f} m median={np.median(distance):.3f} m P95={np.quantile(distance,.95):.3f} m max={np.max(distance):.3f} m mean_XYZ={np.mean(values,axis=0)}")
    return 0

if __name__=="__main__":raise SystemExit(main())
