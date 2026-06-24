#!/usr/bin/env python3
"""Run synchronized rover/base short-baseline PPK."""
from __future__ import annotations
import argparse,copy,csv,hashlib,pickle
from pathlib import Path
import numpy as np
from compass.gnss.ppk import PPKConfig,PPKKalmanFilter
from compass.gnss.precise import PreciseProducts
from compass.gnss.spp import SPPSolver
from compass.gnss.trajectory_smoother import covariance_intersection,trajectory_multipass
from compass.io.rinex_native import RINEXNativeReader

def xyz(value):
    a=np.asarray([float(v) for v in value.replace(',',' ').split()])
    if len(a)!=3:raise argparse.ArgumentTypeError('expected X Y Z')
    return a

def sat_ids(value):
    return tuple(int(item.upper().removeprefix("G")) for item in value.replace(","," ").split())

def read_obs_cached(reader,path,max_epochs,start_sow,enabled=True):
    if not enabled:return reader.read_obs(path,max_epochs=max_epochs,start_sow=start_sow)
    source=Path(path).resolve();stat=source.stat();cache_dir=Path("data")/".cache"/"obs";cache_dir.mkdir(parents=True,exist_ok=True)
    key=f"v2|{source}|{stat.st_size}|{stat.st_mtime_ns}|{max_epochs}|{start_sow:.3f}"
    cache_path=cache_dir/(hashlib.sha1(key.encode("utf-8")).hexdigest()+".pkl")
    if cache_path.exists():
        with cache_path.open("rb") as stream:return pickle.load(stream)
    observations=reader.read_obs(path,max_epochs=max_epochs,start_sow=start_sow)
    with cache_path.open("wb") as stream:pickle.dump(observations,stream,protocol=pickle.HIGHEST_PROTOCOL)
    return observations

def read_products_cached(sp3_files,clk_files,enabled=True):
    if not enabled:return PreciseProducts.from_files(sp3_files,clk_files)
    cache_dir=Path("data")/".cache"/"products";cache_dir.mkdir(parents=True,exist_ok=True)
    parts=["v1"]
    for path in list(sp3_files)+list(clk_files):
        source=Path(path).resolve();stat=source.stat()
        parts.append(f"{source}|{stat.st_size}|{stat.st_mtime_ns}")
    cache_path=cache_dir/(hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()+".pkl")
    if cache_path.exists():
        with cache_path.open("rb") as stream:return pickle.load(stream)
    products=PreciseProducts.from_files(sp3_files,clk_files)
    with cache_path.open("wb") as stream:pickle.dump(products,stream,protocol=pickle.HIGHEST_PROTOCOL)
    return products

def main():
    p=argparse.ArgumentParser();p.add_argument('--rover',required=True);p.add_argument('--base',required=True);p.add_argument('--nav',required=True);p.add_argument('--sp3',required=True,nargs='+');p.add_argument('--clk',nargs='*',default=[]);p.add_argument('--base-position',type=xyz,required=True);p.add_argument('--out',default=r'data\ppk_result.csv');p.add_argument('--start-sow',type=float,default=0.0);p.add_argument('--max-epochs',type=int,default=0);p.add_argument('--elev',type=float,default=15.0);p.add_argument('--acceleration-noise',type=float,default=0.2);p.add_argument('--code-sigma',type=float,default=0.3);p.add_argument('--phase-sigma',type=float,default=0.003);p.add_argument('--ar-ratio',type=float,default=2.5);p.add_argument('--ar-min-epochs',type=int,default=10);p.add_argument('--wide-lane-min-epochs',type=int,default=30);p.add_argument('--ar-strategy',choices=('staged','all'),default='all');p.add_argument('--ionosphere-mode',choices=('estimate','broadcast','off'),default='broadcast');p.add_argument('--ephemeris-mode',choices=('broadcast','precise'),default='broadcast');p.add_argument('--motion-model',choices=('kinematic','continuous'),default='kinematic');p.add_argument('--exclude-sats',type=sat_ids,default=());p.add_argument('--ambiguity-hold-epochs',type=int,default=20);p.add_argument('--wide-lane-hold-sigma',type=float,default=0.5);p.add_argument('--ambiguity-hold-feedback',action=argparse.BooleanOptionalAction,default=False);p.add_argument('--bidirectional-min-age',type=int,default=60);p.add_argument('--filter-mode',choices=('forward','reverse','combined','multipass','bidirectional'),default='forward');p.add_argument('--progress-every',type=int,default=100);p.add_argument('--no-obs-cache',action='store_true');p.add_argument('--product-cache',action='store_true');args=p.parse_args()
    reader=RINEXNativeReader();nav=reader.read_nav(args.nav);read_limit=args.max_epochs if args.max_epochs else 0;base_limit=read_limit if args.start_sow>0.0 else 0;cache_enabled=not args.no_obs_cache;rover=read_obs_cached(reader,args.rover,read_limit,args.start_sow,cache_enabled);base=read_obs_cached(reader,args.base,base_limit,args.start_sow,cache_enabled);base_by_time={(e.week,round(e.timestamp,3)):e for e in base};pairs=[(r,base_by_time[(r.week,round(r.timestamp,3))]) for r in rover if r.timestamp>=args.start_sow and (r.week,round(r.timestamp,3)) in base_by_time]
    if args.max_epochs:pairs=pairs[:args.max_epochs]
    products=read_products_cached(args.sp3,args.clk,args.product_cache);spp=SPPSolver(elev_mask=args.elev)
    def run_pass(direction,label):
        flt=PPKKalmanFilter(products,args.base_position,PPKConfig(elevation_mask_deg=args.elev,acceleration_noise_mps2=args.acceleration_noise,code_sigma_m=args.code_sigma,phase_sigma_m=args.phase_sigma,ratio_threshold=args.ar_ratio,ambiguity_min_epochs=args.ar_min_epochs,wide_lane_min_epochs=args.wide_lane_min_epochs,ambiguity_strategy=args.ar_strategy,time_direction=direction,ionosphere_mode=args.ionosphere_mode,ephemeris_mode=args.ephemeris_mode,motion_model=args.motion_model,excluded_satellites=args.exclude_sats,ambiguity_hold_epochs=args.ambiguity_hold_epochs,wide_lane_hold_sigma_cycles=args.wide_lane_hold_sigma,ambiguity_hold_feedback=args.ambiguity_hold_feedback),reader.nav_ion_gps,reader.nav_ion_bds,navigation=nav);ordered=list(reversed(pairs)) if direction<0 else pairs;rows=[];fixed=success=0;initial=None
        for i,(r,b) in enumerate(ordered,1):
            if initial is None or args.motion_model=="kinematic":
                seed=r.approx_position if initial is None else initial
                candidate,_,ok=spp.solve(r,nav,seed)
                if ok:initial=candidate
                elif initial is None:initial=r.approx_position
            sol=flt.process(r,b,initial);success+=sol.status;fixed+=sol.fix_status;rows.append((r,sol))
            if args.progress_every and (i%args.progress_every==0 or i==len(ordered)):print(f'{label} {i}/{len(ordered)} success={success} fixed={fixed} ns={sol.satellites} ratio={sol.ratio:.2f}',flush=True)
        if direction<0:rows.reverse()
        return rows,success,fixed,flt
    mode='combined' if args.filter_mode=='bidirectional' else args.filter_mode
    if mode=='forward':rows,success,fixed,flt=run_pass(1,'FORWARD')
    elif mode=='reverse':rows,success,fixed,flt=run_pass(-1,'REVERSE')
    else:
        forward,_,_,flt=run_pass(1,'FORWARD');reverse,_,_,_=run_pass(-1,'REVERSE');rows=[]
        for pass_index,((epoch,first),(_,second)) in enumerate(zip(forward,reverse)):
            forward_age=pass_index+1;reverse_age=len(forward)-pass_index
            if first.fix_status:
                sol=copy.deepcopy(first)
                if second.fix_status and reverse_age>=args.bidirectional_min_age and first.integer_signature==second.integer_signature:
                    state,covariance=covariance_intersection(np.r_[first.position,first.velocity],first.covariance,np.r_[second.position,second.velocity],second.covariance)
                    sol.position=state[:3];sol.velocity=state[3:6];sol.covariance=covariance
            elif second.fix_status and reverse_age>=args.bidirectional_min_age:sol=copy.deepcopy(second)
            elif first.status and second.status and forward_age>=args.bidirectional_min_age and reverse_age>=args.bidirectional_min_age:
                sol=copy.deepcopy(first);state,covariance=covariance_intersection(np.r_[first.position,first.velocity],first.covariance,np.r_[second.position,second.velocity],second.covariance)
                sol.position=state[:3];sol.velocity=state[3:6];sol.covariance=covariance;sol.status=1
            else:sol=copy.deepcopy(first if first.status or not second.status else second)
            rows.append((epoch,sol))
        success=sum(sol.status for _,sol in rows);fixed=sum(sol.fix_status for _,sol in rows)
        if mode=='multipass':
            times=[epoch.week*604800.0+epoch.timestamp for epoch,_ in rows]
            states=[np.r_[sol.position,sol.velocity,np.zeros(3)] for _,sol in rows]
            covariances=[]
            for _,sol in rows:
                covariance=np.eye(9)*100.0
                if sol.covariance.shape==(6,6):covariance[:6,:6]=sol.covariance
                covariances.append(covariance)
            smoothed,smoothed_covariances=trajectory_multipass(times,states,covariances,[bool(sol.status) for _,sol in rows],jerk_noise=args.acceleration_noise)
            for item,state,covariance in zip(rows,smoothed,smoothed_covariances):
                sol=item[1]
                if sol.status and not sol.fix_status:sol.position=state[:3];sol.velocity=state[3:6];sol.covariance=covariance[:6,:6]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,'w',newline='',encoding='utf-8') as stream:
        out=csv.writer(stream);out.writerow(['week','sow','x_m','y_m','z_m','vx_mps','vy_mps','vz_mps','sdx_m','sdy_m','sdz_m','status','fix_status','ns','ambiguities','ratio','filter_mode','fix_shift_m','postfit_code_rms_m','postfit_phase_rms_m','used_held','integer_signature'])
        for epoch,sol in rows:
            sd=np.sqrt(np.maximum(np.diag(sol.covariance),0)) if sol.covariance.size else np.full(6,np.nan);out.writerow([epoch.week,f'{epoch.timestamp:.3f}',*(f'{v:.4f}' for v in sol.position),*(f'{v:.5f}' for v in sol.velocity),*(f'{v:.4f}' for v in sd[:3]),sol.status,sol.fix_status,sol.satellites,sol.ambiguities,f'{sol.ratio:.3f}',mode,f'{sol.fix_shift_m:.4f}',f'{sol.postfit_code_rms_m:.4f}',f'{sol.postfit_phase_rms_m:.4f}',sol.used_held,sol.integer_signature])
    print(f'Finished {len(pairs)} synchronized epochs; success={success}; fixed={fixed}; output={args.out}')
    if flt.x is not None and flt.ambiguity_index:
        values=np.asarray([flt.x[flt.ambiguity_index[key]] for key in sorted(flt.ambiguity_index)])
        sigmas=np.asarray([np.sqrt(max(flt.P[flt.ambiguity_index[key],flt.ambiguity_index[key]],0.0)) for key in sorted(flt.ambiguity_index)])
        fractions=np.abs(values-np.rint(values))
        print(f'Ambiguity diagnostics: fraction_rms={np.sqrt(np.mean(fractions**2)):.3f} max_fraction={np.max(fractions):.3f} sigma_median={np.median(sigmas):.3f} cycles')
        print(f'Last fix candidate: count={flt.last_fix_count} position_shift={flt.last_fix_shift:.3f} m')
if __name__=='__main__':main()
