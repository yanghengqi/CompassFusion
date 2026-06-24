#!/usr/bin/env python3
"""Run full-model kinematic IF-float PPP and emit loose-coupling P/V covariance."""
from __future__ import annotations
import argparse, copy, os, sys, time
import numpy as np
from compass.core.transforms import ecef2llh
from compass.gnss.ppp import PPPConfig, PPPKalmanFilter
from compass.gnss.precise import PreciseProducts
from compass.gnss.ppp_models import AntexCalibration, EarthRotationParameters, OceanLoading
from compass.gnss.bias_osb import OsbCalibration
from compass.gnss.orbex_attitude import OrbexAttitude
from compass.gnss.code_dcb import MonthlyCodeDcb
from compass.gnss.trajectory_smoother import RTSSnapshot, covariance_intersection, rts_smooth, trajectory_multipass
from compass.io.rinex_native import RINEXNativeReader


def _xyz(value):
    a=np.asarray([float(v) for v in value.replace(',',' ').split()])
    if a.size!=3: raise argparse.ArgumentTypeError('expected ECEF X,Y,Z metres')
    return a


def _load_optional(path: str, loader):
    if not path or not os.path.isfile(path):
        return None
    try:
        return loader(path)
    except (OSError, ValueError) as exc:
        print(f'Warning: could not load {path}: {exc}', file=sys.stderr)
        return None


def main(argv=None):
    p=argparse.ArgumentParser(description='COMPASS Python full-model float PPP')
    p.add_argument('--obs',required=True);p.add_argument('--nav',required=True)
    p.add_argument('--sp3',required=True,nargs='+');p.add_argument('--clk',nargs='*',default=[])
    p.add_argument('--atx',default=r'data\atx\igs20.atx')
    p.add_argument('--erp',default=r'data\erp\COD0MGXFIN_20211960000_03D_12H_ERP.ERP')
    p.add_argument('--bia',default=r'data\bia\COD0MGXFIN_20211960000_01D_01D_OSB.BIA')
    p.add_argument('--obx',default=r'data\bia\COD0MGXFIN_20211960000_01D_15M_ATT.OBX')
    p.add_argument('--dcb-dir',default=r'data\dcb',help='directory with CODE monthly P1C1/P2C2 DCB files')
    p.add_argument('--receiver-antenna',default='',help='receiver antenna type (default: RINEX header)')
    p.add_argument('--antdel',type=_xyz,help='receiver antenna delta E,N,U in metres')
    p.add_argument('--blq',default='',help='BLQ ocean-loading file')
    p.add_argument('--station',default='',help='four-character station id for BLQ lookup')
    p.add_argument('--out',default=r'data\ppp_result.csv')
    p.add_argument('--approx',type=_xyz);p.add_argument('--elev',type=float,default=10.0)
    p.add_argument('--static',action='store_true',help='use GREAT-style static position model')
    p.add_argument('--ar',action='store_true',help='enable OSB-based PPP ambiguity resolution')
    p.add_argument('--filter-mode',choices=('forward','reverse','combined','multipass','bidirectional'),default='forward',
                   help='forward, reverse, covariance-intersection combined, or multipass RTS')
    p.add_argument('--systems',default='GEC')
    p.add_argument('--max-epochs',type=int,default=0);p.add_argument('--progress-every',type=int,default=100)
    args=p.parse_args(argv);started=time.perf_counter()
    reader=RINEXNativeReader();nav=reader.read_nav(args.nav)
    obs=reader.read_obs(args.obs,max_epochs=max(0,args.max_epochs))
    products=PreciseProducts.from_files(args.sp3,args.clk)
    antex=_load_optional(args.atx, AntexCalibration.from_file)
    erp=_load_optional(args.erp, EarthRotationParameters.from_file)
    osb=_load_optional(args.bia, OsbCalibration.from_file)
    obx=_load_optional(args.obx, OrbexAttitude.from_file)
    dcb=MonthlyCodeDcb.from_directory(args.dcb_dir) if args.dcb_dir and os.path.isdir(args.dcb_dir) else None
    station=args.station or os.path.basename(args.obs)[:4]
    ocean=_load_optional(args.blq, lambda path: OceanLoading.from_file(path, station))
    if dcb and not (dcb.p1c1_m or dcb.p2c2_m):
        dcb = None
    systems=set(args.systems.upper())
    for epoch in obs:epoch.observations=[o for o in epoch.observations if o.system in systems]
    def run_pass(direction,label):
        flt=PPPKalmanFilter(
            products,nav,PPPConfig(elevation_mask_deg=args.elev,static_mode=args.static,ambiguity_resolution=args.ar,time_direction=direction),antex,erp,
            args.receiver_antenna or reader.obs_receiver_antenna,osb=osb,obx=obx,dcb=dcb,
            antdel=args.antdel if args.antdel is not None else reader.obs_antenna_delta_enu,ocean_loading=ocean,
        )
        ordered=list(reversed(obs)) if direction<0 else obs;solutions=[];snapshots=[];success=0;last=None
        for i,epoch in enumerate(ordered,1):
            sol=flt.process(epoch,args.approx);last=sol;success+=sol.status;solutions.append(sol)
            state=np.r_[sol.position,sol.velocity,sol.acceleration];covariance=sol.covariance.copy() if sol.covariance.shape==(9,9) else np.eye(9)*1e12
            snapshots.append(RTSSnapshot(state,covariance,None if flt.smoothing_predicted_state is None else flt.smoothing_predicted_state.copy(),None if flt.smoothing_predicted_covariance is None else flt.smoothing_predicted_covariance.copy(),None if flt.smoothing_transition is None else flt.smoothing_transition.copy(),bool(sol.status),bool(flt.smoothing_transition_valid)))
            if args.progress_every and (i%args.progress_every==0 or i==len(ordered)):
                print(f'{label} {i}/{len(ordered)} success={success} ns={sol.satellites} phase={sol.phase_rms_m:.3f}m code={sol.code_rms_m:.2f}m',flush=True)
        if direction<0:solutions.reverse();snapshots.reverse()
        return solutions,snapshots,success,last
    mode='combined' if args.filter_mode=='bidirectional' else args.filter_mode
    if mode=='forward':solutions,snapshots,success,last=run_pass(1,'FORWARD')
    elif mode=='reverse':solutions,snapshots,success,last=run_pass(-1,'REVERSE')
    else:
        forward,forward_snapshots,_,last=run_pass(1,'FORWARD');reverse,_,_,_=run_pass(-1,'REVERSE');solutions=[]
        for first,second in zip(forward,reverse):
            base=copy.deepcopy(first if first.status or not second.status else second)
            if first.status and second.status:
                state,covariance=covariance_intersection(np.r_[first.position,first.velocity,first.acceleration],first.covariance,np.r_[second.position,second.velocity,second.acceleration],second.covariance)
                base.position=state[:3];base.velocity=state[3:6];base.acceleration=state[6:9];base.covariance=covariance;base.status=1
            solutions.append(base)
        success=sum(sol.status for sol in solutions);snapshots=forward_snapshots
        if mode=='multipass':
            states,covariances=trajectory_multipass([epoch.week*604800.0+epoch.timestamp for epoch in obs],[np.r_[sol.position,sol.velocity,sol.acceleration] for sol in solutions],[sol.covariance if sol.covariance.shape==(9,9) else np.eye(9)*1e12 for sol in solutions],[bool(sol.status) for sol in solutions])
            for sol,state,covariance in zip(solutions,states,covariances):
                if sol.status:sol.position=state[:3];sol.velocity=state[3:6];sol.acceleration=state[6:9];sol.covariance=covariance
    with open(args.out,'w',encoding='utf-8',newline='\n') as out:
        out.write('week,sow,x_m,y_m,z_m,lat_deg,lon_deg,h_m,vx_mps,vy_mps,vz_mps,ax_mps2,ay_mps2,az_mps2,zwd_m,sdx_m,sdy_m,sdz_m,sdvx_mps,sdvy_mps,sdvz_mps,ns,nmeas,rejected,phase_rms_m,code_rms_m,doppler_rms_mps,status,fix_status,fixed_ambiguities,ar_ratio,filter_mode\n')
        for epoch,sol in zip(obs,solutions):
            llh=ecef2llh(sol.position) if np.all(np.isfinite(sol.position)) else np.full(3,np.nan)
            sd=np.sqrt(np.maximum(np.diag(sol.covariance),0)) if sol.covariance.size else np.full(9,np.nan)
            vals=[epoch.week,f'{epoch.timestamp:.3f}',*(f'{v:.4f}' for v in sol.position),f'{np.degrees(llh[0]):.9f}',f'{np.degrees(llh[1]):.9f}',f'{llh[2]:.4f}',*(f'{v:.5f}' for v in sol.velocity),*(f'{v:.6f}' for v in sol.acceleration),f'{sol.zwd_m:.4f}',*(f'{v:.4f}' for v in sd[:6]),sol.satellites,sol.measurements,sol.rejected_satellites,f'{sol.phase_rms_m:.4f}',f'{sol.code_rms_m:.3f}',f'{sol.doppler_rms_mps:.3f}',sol.status,sol.fix_status,sol.fixed_ambiguities,f'{sol.ar_ratio:.3f}',mode]
            out.write(','.join(map(str,vals))+'\n')
    print(f'Finished {len(obs)} epochs in {time.perf_counter()-started:.1f}s; success={success}/{len(obs)}; output={args.out}')
    if last:
        print('Model status:')
        for k,v in last.model_status.items():print(f'  {k}: {v}')
    return 0

if __name__=='__main__':raise SystemExit(main())
