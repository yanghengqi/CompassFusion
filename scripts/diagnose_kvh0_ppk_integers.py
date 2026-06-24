#!/usr/bin/env python3
"""Audit KVH0 DD integers against the fixed reference trajectory."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from compass.gnss.ppk import F1,F2,L1,L2,WL,PPKConfig,PPKKalmanFilter
from compass.gnss.precise import PreciseProducts
from compass.gnss.spp import SPPSolver
from compass.io.rinex_native import RINEXNativeReader

ROOT=Path(__file__).resolve().parents[1];GAMMA=(F1/F2)**2
def reference(path):
    out={}
    for line in path.open(errors="ignore"):
        p=[v.strip() for v in line.split(",")]
        try:out[float(p[1])]=np.asarray(p[2:5],float)
        except (ValueError,IndexError):pass
    return out
def mw(s):
    p1,p2,n1,n2=s[:4]
    return ((F1*n1*L1-F2*n2*L2)/(F1-F2)-(F1*p1+F2*p2)/(F1+F2))/WL
def clean_mean(values):
    a=np.asarray(values);m=np.median(a);mad=np.median(abs(a-m));a=a[abs(a-m)<=max(.25,6*1.4826*mad)]
    return a.mean(),a.std(),len(a)
def main():
    r=RINEXNativeReader();nav=r.read_nav(ROOT/"data/nav/brdm1960.21p");rov=r.read_obs(ROOT/"data/obs/KVH01960.21o");base=r.read_obs(ROOT/"data/obs/KVH01960_base.21o")
    products=PreciseProducts.from_files([ROOT/"data/sp3/COD0MGXFIN_20211960000_01D_05M_ORB.SP3"],[ROOT/"data/clk/COD0MGXFIN_20211960000_01D_30S_CLK.CLK"])
    base_xyz=np.array([-2409529.6373,4703523.4787,3559037.4588]);truth=reference(ROOT/"data/KVH0_fix.pos")
    base_time={(e.week,round(e.timestamp,3)):e for e in base};pairs=[(e,base_time[(e.week,round(e.timestamp,3))]) for e in rov if e.timestamp in truth and (e.week,round(e.timestamp,3)) in base_time]
    model=PPKKalmanFilter(products,base_xyz,PPKConfig());wide=defaultdict(list);ifbias=defaultdict(list);refs=defaultdict(int)
    for re,be in pairs:
        rx=truth[re.timestamp];t=re.week*604800+re.timestamp;rm={(o.system,o.sat_id):o for o in re.observations};bm={(o.system,o.sat_id):o for o in be.observations};sats={}
        for key,ro in rm.items():
            if key[0]!="G" or key not in bm:continue
            rs=model._signal(ro);bs=model._signal(bm[key])
            if rs is None or bs is None:continue
            sr=model._satellite(key[1],t,rs[0],rx);sb=model._satellite(key[1],t,bs[0],base_xyz)
            if sr is None or sb is None:continue
            el=model._elevation(sr,rx)
            if el>=np.deg2rad(15):sats[key[1]]=(rs,bs,sr,sb,el)
        if len(sats)<5:continue
        ref=max(sats,key=lambda sat:sats[sat][4]);refs[ref]+=1
        def geom(sat):
            _,_,sr,sb,_=sats[sat]
            return np.linalg.norm(sr-rx)-np.linalg.norm(sb-base_xyz)+model._troposphere(sr,rx)-model._troposphere(sb,base_xyz)
        rrs,rbs=sats[ref][:2];rg=geom(ref);rmw=mw(rrs)-mw(rbs)
        for sat,item in sats.items():
            if sat==ref:continue
            rs,bs=item[:2];l1=(rs[2]-bs[2]-rrs[2]+rbs[2])*L1;l2=(rs[3]-bs[3]-rrs[3]+rbs[3])*L2;key=(ref,sat)
            wide[key].append(mw(rs)-mw(bs)-rmw);ifbias[key].append((GAMMA*l1-l2)/(GAMMA-1)-(geom(sat)-rg))
    print(f"epochs={len(pairs)} references={dict(sorted(refs.items()))}");print("ref sat   n  WL(float/int)  N1(float/int)  N2int scatter(WL/IF)")
    for key in sorted(wide):
        w,ws,n=clean_mean(wide[key]);b,bs,_=clean_mean(ifbias[key]);wi=round(w);n1=((GAMMA-1)*b-L2*wi)/(GAMMA*L1-L2);n1i=round(n1)
        print(f"G{key[0]:02d} G{key[1]:02d} {n:3d} {w:8.3f}/{wi:4d} {n1:9.3f}/{n1i:4d} {n1i-wi:6d} {ws:.3f}/{bs:.3f}")
    def run_filter(direction):
        config=PPKConfig(time_direction=direction);flt=PPKKalmanFilter(products,base_xyz,config);spp=SPPSolver(elev_mask=15);initial=None
        ordered=pairs if direction>0 else list(reversed(pairs))
        for re,be in ordered:
            if initial is None:
                initial,_,ok=spp.solve(re,nav,re.approx_position)
                if not ok:initial=re.approx_position
            flt.process(re,be,initial)
        print(f"{'forward' if direction>0 else 'reverse'} filter reference=G{flt.reference:02d}")
        for sat in sorted({sat for _,sat in flt.ambiguity_index}):
            if (1,sat) in flt.ambiguity_index and (2,sat) in flt.ambiguity_index:
                n1=flt.x[flt.ambiguity_index[(1,sat)]];n2=flt.x[flt.ambiguity_index[(2,sat)]]
                print(f"filter G{sat:02d}: N1={n1:.3f} N2={n2:.3f} WL={n1-n2:.3f}")
    run_filter(1);run_filter(-1)
if __name__=="__main__":main()
