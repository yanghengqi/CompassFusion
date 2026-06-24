"""Short-baseline dual-frequency GPS PPK with double-difference ambiguity fixing."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.constants import CLIGHT
from ..core.gnss_types import GNSSRawObservation, SatelliteObservation
from ..core.transforms import ecef2llh
from .ionosphere import ionocorr
from .lambda_ar import lambda_algorithm
from .precise import PreciseProducts
from .spp import SPPSolver

OMGE=7.2921151467e-5
F1,F2=1575.42e6,1227.60e6
L1,L2=CLIGHT/F1,CLIGHT/F2
WL=CLIGHT/(F1-F2)


@dataclass
class PPKConfig:
    elevation_mask_deg: float = 15.0
    min_satellites: int = 5
    acceleration_noise_mps2: float = 2.0
    code_sigma_m: float = 0.5
    phase_sigma_m: float = 0.01
    ratio_threshold: float = 2.5
    ambiguity_constraint_sigma_cycles: float = 1.0e-3
    gf_slip_threshold_m: float = 0.08
    ionosphere_noise_m_sqrt_s: float = 1.0e-4
    initial_ionosphere_sigma_m: float = 0.02
    wide_lane_min_epochs: int = 30
    wide_lane_fraction: float = 0.25
    ambiguity_min_epochs: int = 60
    min_fix_satellites: int = 4
    ambiguity_strategy: str = "staged"
    time_direction: int = 1
    ionosphere_mode: str = "estimate"
    ephemeris_mode: str = "precise"
    motion_model: str = "continuous"
    excluded_satellites: tuple[int, ...] = ()
    ambiguity_hold_epochs: int = 20
    wide_lane_hold_sigma_cycles: float = 0.5
    ambiguity_hold_feedback: bool = True


@dataclass
class PPKSolution:
    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    status: int
    fix_status: int
    satellites: int
    ambiguities: int
    ratio: float
    integer_signature: str = ""
    fix_shift_m: float = math.nan
    postfit_code_rms_m: float = math.nan
    postfit_phase_rms_m: float = math.nan
    used_held: int = 0


class PPKKalmanFilter:
    def __init__(self,products:PreciseProducts,base_position:np.ndarray,config:Optional[PPKConfig]=None,ion_gps:Optional[np.ndarray]=None,ion_bds:Optional[np.ndarray]=None,navigation=None):
        self.products=products;self.base=np.asarray(base_position,float).reshape(3);self.config=config or PPKConfig();self.navigation=navigation or [];self.broadcast_solver=SPPSolver(elev_mask=self.config.elevation_mask_deg)
        self.ion_gps=None if ion_gps is None else np.asarray(ion_gps,float).reshape(-1)[:8]
        self.ion_bds=None if ion_bds is None else np.asarray(ion_bds,float).reshape(-1)[:8]
        self.x:Optional[np.ndarray]=None;self.P:Optional[np.ndarray]=None;self.last_time:Optional[float]=None
        self.reference:Optional[int]=None;self.ambiguity_index:Dict[Tuple[int,int],int]={};self.ambiguity_age:Dict[Tuple[int,int],int]={};self.ionosphere_index:Dict[int,int]={};self.gf_sd={};self.mw_mean={};self.mw_m2={};self.mw_count={};self.wide_lane_integer={}
        self.last_fix_shift=np.nan;self.last_fix_count=0;self.last_fixed_signature="";self.last_postfit_code_rms=np.nan;self.last_postfit_phase_rms=np.nan;self.last_used_held=0;self.integer_votes={};self.held_integers={};self.unstable_ambiguity_satellites=set();self.last_integer_candidate={}

    @staticmethod
    def _signal(ob:SatelliteObservation):
        raw=ob.raw_observations or {}
        second="2W" if all(k in raw for k in ("C2W","L2W")) else "2X"
        names=("C1C","C"+second,"L1C","L"+second)
        if not all(k in raw and raw[k][0] not in (None,0.0) for k in names):return None
        return tuple(float(raw[k][0]) for k in names)+(int(raw["L1C"][1]),int(raw["L"+second][1]))

    @staticmethod
    def _rotate(pos,travel):
        a=OMGE*travel;c,s=math.cos(a),math.sin(a)
        return np.array([[c,s,0],[-s,c,0],[0,0,1]])@pos

    def _satellite_state(self,sat_id,time,code,receiver):
        transmit_time=time-code/CLIGHT
        if self.config.ephemeris_mode=="broadcast":
            nav=self.broadcast_solver._find_nav(sat_id,"G",transmit_time,self.navigation)
            if nav is None:return None
            _,_,clock=self.broadcast_solver._compute_sat_pos(nav,transmit_time)
            transmit_time-=clock
            pos,_,clock=self.broadcast_solver._compute_sat_pos(nav,transmit_time)
        else:
            state=self.products.state_full("G",sat_id,transmit_time)
            if state is None:return None
            transmit_time-=state[2]
            state=self.products.state_full("G",sat_id,transmit_time)
            if state is None:return None
            pos,clock=state[0],state[2]
        travel=np.linalg.norm(pos-receiver)/CLIGHT
        return self._rotate(pos,travel),clock

    def _satellite(self,sat_id,time,code,receiver):
        state=self._satellite_state(sat_id,time,code,receiver)
        return None if state is None else state[0]

    @staticmethod
    def _elevation(sat,rec):
        llh=ecef2llh(rec);u=(sat-rec)/np.linalg.norm(sat-rec);sl,cl=math.sin(llh[1]),math.cos(llh[1]);sp,cp=math.sin(llh[0]),math.cos(llh[0])
        up=np.array([cp*cl,cp*sl,sp]);return math.asin(np.clip(np.dot(u,up),-1.0,1.0))

    @staticmethod
    def _azel(sat,rec):
        llh=ecef2llh(rec);u=(sat-rec)/np.linalg.norm(sat-rec);sl,cl=math.sin(llh[1]),math.cos(llh[1]);sp,cp=math.sin(llh[0]),math.cos(llh[0])
        enu=np.array([[-sl,cl,0],[-sp*cl,-sp*sl,cp],[cp*cl,cp*sl,sp]])@u
        return math.atan2(enu[0],enu[1])%(2*math.pi),math.asin(np.clip(enu[2],-1.0,1.0))

    def _ionosphere_l1(self,sat,rec,time):
        if self.config.ionosphere_mode!="broadcast":return 0.0
        az,el=self._azel(sat,rec);llh=ecef2llh(rec)
        return ionocorr(time%604800.0,self.ion_gps,self.ion_bds,llh,np.array([az,el]),"G")[0]

    @classmethod
    def _troposphere(cls,sat,rec):
        llh=ecef2llh(rec);height=max(-1000.0,min(10000.0,float(llh[2])));elevation=cls._elevation(sat,rec)
        pressure=1013.25*(1.0-2.2557e-5*height)**5.2568
        zhd=0.0022768*pressure/(1.0-0.00266*math.cos(2.0*llh[0])-0.00028*height/1000.0)
        mapping=1.0/max(math.sin(elevation),0.15)
        return (zhd+0.12)*mapping

    def _reset_ambiguities(self):
        if self.x is not None:self.x=self.x[:6].copy();self.P=self.P[:6,:6].copy()
        self.ambiguity_index.clear();self.ambiguity_age.clear();self.ionosphere_index.clear();self.gf_sd.clear();self.mw_mean.clear();self.mw_m2.clear();self.mw_count.clear();self.wide_lane_integer.clear();self.integer_votes.clear();self.held_integers.clear();self.unstable_ambiguity_satellites.clear();self.last_integer_candidate.clear()

    def _add_ambiguity(self,key,value):
        i=len(self.x);self.ambiguity_index[key]=i;self.ambiguity_age[key]=1;self.x=np.append(self.x,value)
        covariance=np.zeros((i+1,i+1));covariance[:i,:i]=self.P;covariance[i,i]=100.0**2;self.P=covariance
        return i

    def _add_ionosphere(self,sat):
        i=len(self.x);self.ionosphere_index[sat]=i;self.x=np.append(self.x,0.0)
        covariance=np.zeros((i+1,i+1));covariance[:i,:i]=self.P;covariance[i,i]=self.config.initial_ionosphere_sigma_m**2;self.P=covariance
        return i

    def _predict(self,time):
        if self.last_time is None:return
        dt=time-self.last_time;h=abs(dt)
        if dt*self.config.time_direction<=0.0 or h>30.0:self._reset_ambiguities();self.last_time=time;return
        if self.config.motion_model=="kinematic":
            for index in self.ionosphere_index.values():self.P[index,index]+=self.config.ionosphere_noise_m_sqrt_s**2*h
            self.last_time=time;return
        n=len(self.x);F=np.eye(n);F[:3,3:6]=np.eye(3)*dt;q=self.config.acceleration_noise_mps2**2
        Q=np.zeros((n,n));direction=1.0 if dt>0.0 else -1.0;block=q*np.array([[h**3/3,direction*h**2/2],[direction*h**2/2,h]])
        for axis in range(3):Q[np.ix_([axis,axis+3],[axis,axis+3])]=block
        for index in self.ionosphere_index.values():Q[index,index]=self.config.ionosphere_noise_m_sqrt_s**2*h
        self.x=F@self.x;self.P=F@self.P@F.T+Q;self.last_time=time

    def process(self,rover:GNSSRawObservation,base:GNSSRawObservation,initial_position:Optional[np.ndarray]=None)->PPKSolution:
        time=rover.week*604800.0+rover.timestamp
        if self.x is None:
            if initial_position is None:return self._empty()
            self.x=np.r_[np.asarray(initial_position,float),np.zeros(3)];self.P=np.diag([25.0]*3+[4.0]*3);self.last_time=time
        else:
            self._predict(time)
            if self.config.motion_model=="kinematic" and initial_position is not None:
                self.x[:3]=np.asarray(initial_position,float);self.x[3:6]=0.0;self.P[:6,:]=0.0;self.P[:,:6]=0.0
                self.P[:3,:3]=np.eye(3)*30.0**2;self.P[3:6,3:6]=np.eye(3)*100.0**2
        rover_map={(o.system,o.sat_id):o for o in rover.observations};base_map={(o.system,o.sat_id):o for o in base.observations}
        satellites={}
        for key,ro in rover_map.items():
            if key[0]!="G" or key not in base_map or key[1] in self.config.excluded_satellites:continue
            br=base_map[key];rs=self._signal(ro);bs=self._signal(br)
            if rs is None or bs is None:continue
            rover_state=self._satellite_state(key[1],time,rs[0],self.x[:3]);base_state=self._satellite_state(key[1],time,bs[0],self.base)
            if rover_state is None or base_state is None:continue
            sat_rover,clock_rover=rover_state;sat_base,clock_base=base_state;elevation=self._elevation(sat_rover,self.x[:3])
            if elevation<math.radians(self.config.elevation_mask_deg):continue
            satellites[key[1]]=(ro,br,rs,bs,sat_rover,sat_base,elevation,clock_rover-clock_base)
        if len(satellites)<self.config.min_satellites:return self._solution(0,0,len(satellites),0.0)
        if self.reference not in satellites:
            self.reference=max(satellites,key=lambda sat:satellites[sat][6]);self._reset_ambiguities()
        ref=self.reference;rr=self.x[:3]
        def geometry(item):
            sat_rover,sat_base=item[4],item[5]
            single_difference=np.linalg.norm(sat_rover-rr)-np.linalg.norm(sat_base-self.base)-CLIGHT*item[7]+self._troposphere(sat_rover,rr)-self._troposphere(sat_base,self.base)
            ionosphere=self._ionosphere_l1(sat_rover,rr,time)-self._ionosphere_l1(sat_base,self.base,time)
            return single_difference,(rr-sat_rover)/np.linalg.norm(rr-sat_rover),ionosphere
        ref_geom,ref_los,ref_iono=geometry(satellites[ref]);ref_rs,ref_bs=satellites[ref][2],satellites[ref][3]
        def mw_cycles(signal):
            p1,p2,l1_cycles,l2_cycles=signal[:4];l1_m=l1_cycles*L1;l2_m=l2_cycles*L2
            return ((F1*l1_m-F2*l2_m)/(F1-F2)-(F1*p1+F2*p2)/(F1+F2))/WL
        ref_slip=bool(ref_rs[4] or ref_rs[5] or ref_bs[4] or ref_bs[5])
        if ref_slip:self.mw_mean.pop(ref,None);self.mw_m2.pop(ref,None);self.mw_count.pop(ref,None)
        ref_mw=mw_cycles(ref_rs)-mw_cycles(ref_bs);count=self.mw_count.get(ref,0)+1;mean=self.mw_mean.get(ref,0.0);delta=ref_mw-mean;mean+=delta/count
        self.mw_count[ref]=count;self.mw_mean[ref]=mean;self.mw_m2[ref]=self.mw_m2.get(ref,0.0)+delta*(ref_mw-mean)
        rows=[];residuals=[];variances=[];row_sats=[];row_types=[]
        for sat,item in sorted(satellites.items()):
            if sat==ref:continue
            ro,br,rs,bs,_,_,elevation,_=item;geom,los,sd_iono=geometry(item);dd_geom=geom-ref_geom;hpos=los-ref_los;dd_iono=sd_iono-ref_iono
            estimate_ionosphere=self.config.ionosphere_mode=="estimate"
            ionosphere_index=None
            if estimate_ionosphere:
                ionosphere_index=self.ionosphere_index.get(sat)
                if ionosphere_index is None:ionosphere_index=self._add_ionosphere(sat)
            gf=(rs[2]*L1-rs[3]*L2)-(bs[2]*L1-bs[3]*L2);slip=bool(rs[4] or rs[5] or bs[4] or bs[5])
            if sat in self.gf_sd and abs(gf-self.gf_sd[sat])>self.config.gf_slip_threshold_m:slip=True
            self.gf_sd[sat]=gf
            mw_sd=mw_cycles(rs)-mw_cycles(bs)
            if slip:
                self.mw_mean.pop(sat,None);self.mw_m2.pop(sat,None);self.mw_count.pop(sat,None)
                for frequency in (1,2):self.integer_votes.pop((frequency,sat),None);self.held_integers.pop((frequency,sat),None)
            count=self.mw_count.get(sat,0)+1;mean=self.mw_mean.get(sat,0.0);delta=mw_sd-mean;mean+=delta/count
            self.mw_count[sat]=count;self.mw_mean[sat]=mean;self.mw_m2[sat]=self.mw_m2.get(sat,0.0)+delta*(mw_sd-mean)
            observations=((rs[0]-bs[0])-(ref_rs[0]-ref_bs[0]),(rs[1]-bs[1])-(ref_rs[1]-ref_bs[1]),(rs[2]*L1-bs[2]*L1)-(ref_rs[2]*L1-ref_bs[2]*L1),(rs[3]*L2-bs[3]*L2)-(ref_rs[3]*L2-ref_bs[3]*L2))
            for frequency,(observed,wavelength) in enumerate(((observations[2],L1),(observations[3],L2)),1):
                key=(frequency,sat)
                if slip and key in self.ambiguity_index:
                    i=self.ambiguity_index[key];self.x[i]=(observed-dd_geom)/wavelength;self.P[i,:]=0;self.P[:,i]=0;self.P[i,i]=100.0**2;self.ambiguity_age[key]=1
                elif key not in self.ambiguity_index:self._add_ambiguity(key,(observed-dd_geom)/wavelength)
                else:self.ambiguity_age[key]=self.ambiguity_age.get(key,0)+1
            sinel=max(math.sin(elevation),0.2)
            gamma=(F1/F2)**2;iono=self.x[ionosphere_index] if estimate_ionosphere else dd_iono
            for observation_type,(observed,factor) in enumerate(((observations[0],1.0),(observations[1],gamma))):
                row=np.zeros(len(self.x));row[:3]=hpos
                if estimate_ionosphere:row[ionosphere_index]=factor
                rows.append(row);residuals.append(observed-dd_geom-factor*iono);variances.append((self.config.code_sigma_m/sinel)**2*2);row_sats.append(sat);row_types.append(observation_type)
            for frequency,(observed,wavelength,factor) in enumerate(((observations[2],L1,1.0),(observations[3],L2,gamma)),1):
                row=np.zeros(len(self.x));row[:3]=hpos
                if estimate_ionosphere:row[ionosphere_index]=-factor
                row[self.ambiguity_index[(frequency,sat)]]=wavelength
                rows.append(row);residuals.append(observed-dd_geom+factor*iono-wavelength*self.x[self.ambiguity_index[(frequency,sat)]]);variances.append((self.config.phase_sigma_m/sinel)**2*2);row_sats.append(sat);row_types.append(frequency+1)
        if not rows:return self._solution(0,0,len(satellites),0.0)
        H=np.zeros((len(rows),len(self.x)))
        for row_index,row in enumerate(rows):H[row_index,:len(row)]=row
        v=np.asarray(residuals);R=np.zeros((len(rows),len(rows)));ref_sinel=max(math.sin(satellites[ref][6]),0.2)
        reference_variances=np.array([2*(self.config.code_sigma_m/ref_sinel)**2]*2+[2*(self.config.phase_sigma_m/ref_sinel)**2]*2)
        for i,observation_type in enumerate(row_types):
            R[i,i]=variances[i]+reference_variances[observation_type]
            for j in range(i):
                if row_types[j]==observation_type:R[i,j]=R[j,i]=reference_variances[observation_type]
        update_origin=self.x.copy()
        S=H@self.P@H.T+R
        try:K=np.linalg.solve(S,H@self.P).T
        except np.linalg.LinAlgError:return self._solution(0,0,len(satellites),0.0)
        self.x+=K@v;A=np.eye(len(self.x))-K@H;self.P=A@self.P@A.T+K@R@K.T;self.P=(self.P+self.P.T)/2
        ambiguity_x,ambiguity_P=self._apply_wide_lane_constraints(ref)
        fix,ratio,fixed_x,fixed_P=self._fix_ambiguities(ambiguity_x,ambiguity_P)
        residual_state=fixed_x if fix and fixed_x is not None else self.x
        postfit=v-H@(residual_state-update_origin)
        code=[postfit[i] for i,t in enumerate(row_types) if t in (0,1)]
        phase=[postfit[i] for i,t in enumerate(row_types) if t in (2,3)]
        self.last_postfit_code_rms=float(np.sqrt(np.mean(np.square(code)))) if code else np.nan
        self.last_postfit_phase_rms=float(np.sqrt(np.mean(np.square(phase)))) if phase else np.nan
        return self._solution(1,fix,len(satellites),ratio,fixed_x,fixed_P)

    def _fix_ambiguities(self,float_x=None,float_P=None):
        self.last_fixed_signature="";self.last_fix_shift=np.nan;self.last_fix_count=0;self.last_used_held=0
        float_x=self.x if float_x is None else float_x;float_P=self.P if float_P is None else float_P
        satellites=sorted({sat for _,sat in self.ambiguity_index if self.ambiguity_age.get((1,sat),0)>=self.config.ambiguity_min_epochs and self.ambiguity_age.get((2,sat),0)>=self.config.ambiguity_min_epochs})
        if self.config.ambiguity_strategy=="staged":satellites=[sat for sat in satellites if sat in self.wide_lane_integer]
        satellites.sort(key=lambda sat:max(float_P[self.ambiguity_index[(frequency,sat)],self.ambiguity_index[(frequency,sat)]] for frequency in (1,2)))
        accepted=None;ratio=0.0
        for count in range(len(satellites),self.config.min_fix_satellites-1,-1):
            subset=satellites[:count]
            if self.config.ambiguity_strategy=="staged":indices=[self.ambiguity_index[(1,sat)] for sat in subset]
            else:indices=[self.ambiguity_index[(frequency,sat)] for sat in subset for frequency in (1,2)]
            afloat=float_x[indices];Q=float_P[np.ix_(indices,indices)];fixed,scores,ok=lambda_algorithm(afloat,Q,2)
            trial_ratio=0.0 if not ok or len(scores)<2 or scores[0]<=1e-12 else float(scores[1]/scores[0]);ratio=max(ratio,trial_ratio)
            if trial_ratio>=self.config.ratio_threshold:
                accepted=(subset,np.rint(fixed[:,0]));ratio=trial_ratio;break
        if accepted is not None:
            candidate_satellites,candidate=accepted;candidate_values={}
            if self.config.ambiguity_strategy=="staged":
                for sat,n1 in zip(candidate_satellites,candidate):
                    candidate_values[(1,sat)]=int(n1);candidate_values[(2,sat)]=int(n1-self.wide_lane_integer[sat])
            else:
                for i,sat in enumerate(candidate_satellites):
                    candidate_values[(1,sat)]=int(candidate[2*i]);candidate_values[(2,sat)]=int(candidate[2*i+1])
            if self.config.ambiguity_hold_epochs>0:
                for key,value in candidate_values.items():
                    sat=key[1];count_sat=self.mw_count.get(sat,0);count_ref=self.mw_count.get(self.reference,0)
                    variance_sat=self.mw_m2.get(sat,0.0)/max(count_sat-1,1);variance_ref=self.mw_m2.get(self.reference,0.0)/max(count_ref-1,1)
                    if count_sat<self.config.ambiguity_hold_epochs or math.sqrt(max(variance_sat+variance_ref,0.0))>self.config.wide_lane_hold_sigma_cycles:continue
                    arc_previous=self.last_integer_candidate.get(key);self.last_integer_candidate[key]=value
                    if arc_previous is not None and arc_previous!=value:
                        self.unstable_ambiguity_satellites.add(sat)
                        for frequency in (1,2):self.held_integers.pop((frequency,sat),None)
                    if sat in self.unstable_ambiguity_satellites:continue
                    previous,count=self.integer_votes.get(key,(None,0))
                    count=count+1 if previous==value else 1;self.integer_votes[key]=(value,count)
                    if count>=self.config.ambiguity_hold_epochs:self.held_integers[key]=value
        held_satellites=sorted(sat for sat in satellites if (1,sat) in self.held_integers and (2,sat) in self.held_integers)
        using_held=len(held_satellites)>=self.config.min_fix_satellites
        held_ratio=0.0
        if using_held:
            held_ratio=self._held_ratio(held_satellites)
            ratio=max(ratio,held_ratio)
            if accepted is None and self.mw_count and held_ratio<1.0:
                using_held=False
        if using_held:
            satellites=held_satellites;target=np.asarray([self.held_integers[(frequency,sat)] for sat in satellites for frequency in (1,2)],float)
            solve_x=self.x;solve_P=self.P;self.last_used_held=1
        elif accepted is not None:
            satellites,candidate=accepted
            if self.config.ambiguity_strategy=="staged":
                target=np.asarray([value for sat,n1 in zip(satellites,candidate) for value in (n1,n1-self.wide_lane_integer[sat])])
            else:target=np.asarray(candidate)
            solve_x=float_x;solve_P=float_P
        else:return 0,ratio,None,None
        rows=[]
        for sat in satellites:
            for frequency in (1,2):
                row=np.zeros(len(self.x));row[self.ambiguity_index[(frequency,sat)]]=1.0;rows.append(row)
        H=np.vstack(rows);self.last_fix_count=len(target)
        self.last_fixed_signature=";".join(f"G{sat:02d}:{int(target[2*i])},{int(target[2*i+1])}" for i,sat in enumerate(satellites))
        R=np.eye(len(H))*self.config.ambiguity_constraint_sigma_cycles**2;S=H@solve_P@H.T+R
        try:K=np.linalg.solve(S,H@solve_P).T
        except np.linalg.LinAlgError:return 0,ratio,None,None
        fixed_x=solve_x+K@(target-H@solve_x)
        self.last_fix_shift=float(np.linalg.norm(fixed_x[:3]-solve_x[:3]))
        if self.last_fix_shift>2.0:return 0,ratio,None,None
        A=np.eye(len(self.x))-K@H;fixed_P=A@solve_P@A.T+K@R@K.T;fixed_P=(fixed_P+fixed_P.T)/2
        if using_held and self.config.ambiguity_hold_feedback:
            for row,value in zip(rows,target):
                index=int(np.flatnonzero(row)[0]);self.x[index]=value;self.P[index,:]=0.0;self.P[:,index]=0.0;self.P[index,index]=self.config.ambiguity_constraint_sigma_cycles**2
        return 1,ratio,fixed_x,fixed_P

    def _apply_wide_lane_constraints(self,ref):
        self.wide_lane_integer.clear()
        if self.mw_count.get(ref,0)<self.config.wide_lane_min_epochs:return self.x,self.P
        constraints=[]
        for sat in sorted(self.mw_mean):
            if sat==ref or self.mw_count.get(sat,0)<self.config.wide_lane_min_epochs:continue
            if (1,sat) not in self.ambiguity_index or (2,sat) not in self.ambiguity_index:continue
            value=self.mw_mean[sat]-self.mw_mean[ref];integer=round(value)
            if abs(value-integer)>self.config.wide_lane_fraction:continue
            row=np.zeros(len(self.x));row[self.ambiguity_index[(1,sat)]]=1.0;row[self.ambiguity_index[(2,sat)]]=-1.0
            constraints.append((row,float(integer)));self.wide_lane_integer[sat]=integer
        if not constraints:return self.x,self.P
        H=np.vstack([item[0] for item in constraints]);target=np.asarray([item[1] for item in constraints]);R=np.eye(len(H))*0.05**2;S=H@self.P@H.T+R
        try:K=np.linalg.solve(S,H@self.P).T
        except np.linalg.LinAlgError:return self.x,self.P
        conditioned_x=self.x+K@(target-H@self.x);A=np.eye(len(self.x))-K@H
        conditioned_P=A@self.P@A.T+K@R@K.T;conditioned_P=(conditioned_P+conditioned_P.T)/2
        return conditioned_x,conditioned_P

    def _held_ratio(self,satellites):
        if not satellites or self.reference is None:return 0.0
        ref_count=self.mw_count.get(self.reference,0)
        if ref_count < self.config.wide_lane_min_epochs:return 0.0
        ref_std=math.sqrt(max(self.mw_m2.get(self.reference,0.0)/max(ref_count-1,1),0.0))
        deltas=[]
        for sat in satellites:
            sat_count=self.mw_count.get(sat,0)
            if sat_count < self.config.wide_lane_min_epochs:continue
            sat_std=math.sqrt(max(self.mw_m2.get(sat,0.0)/max(sat_count-1,1),0.0))
            deltas.append(sat_std+ref_std)
        if not deltas:return 0.0
        return float(np.clip(1.0/max(np.mean(deltas),0.02),0.0,100.0))

    def _solution(self,status,fix,satellites,ratio,state=None,covariance=None):
        state=self.x if state is None else state;covariance=self.P if covariance is None else covariance
        fix_shift=self.last_fix_shift if status else np.nan
        code_rms=self.last_postfit_code_rms if status else np.nan
        phase_rms=self.last_postfit_phase_rms if status else np.nan
        return PPKSolution(state[:3].copy(),state[3:6].copy(),covariance[:6,:6].copy(),status,fix,satellites,len(self.ambiguity_index),ratio,self.last_fixed_signature if fix else "",fix_shift,code_rms,phase_rms,self.last_used_held if fix else 0)

    @staticmethod
    def _empty():
        return PPKSolution(np.full(3,np.nan),np.full(3,np.nan),np.empty((0,0)),0,0,0,0,0.0,"")
