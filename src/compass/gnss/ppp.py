"""Engineering-grade dual-frequency float PPP for loose GNSS/INS coupling."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple
from collections import deque
import math
import numpy as np

from ..core.constants import CLIGHT
from ..core.gnss_types import GNSSRawObservation, NavRecord, SatelliteObservation
from ..core.transforms import ecef2llh
from .precise import PreciseProducts
from .spp import SPPSolver
from .ppp_models import (
    AntexCalibration, EarthRotationParameters, OceanLoading, earth_tide_displacement,
    phase_windup_cycles, sun_moon_ecef,
)
from .bias_osb import OsbCalibration
from .orbex_attitude import OrbexAttitude
from .code_dcb import MonthlyCodeDcb
from .lambda_ar import lambda_algorithm

OMGE = 7.2921151467e-5
GM_EARTH = 3.986004418e14
SYSTEMS = ("G", "R", "E", "C", "J")


@dataclass
class PPPConfig:
    ambiguity_resolution: bool = False
    ar_min_epochs: int = 120
    ar_min_satellites: int = 5
    ar_min_elevation_deg: float = 15.0
    ar_wide_lane_fraction: float = 0.20
    ar_partial_narrow_lane_fraction: float = 0.20
    ar_partial_min_elevation_deg: float = 25.0
    ar_wide_lane_sigma_cycles: float = 0.10
    ar_narrow_lane_sigma_cycles: float = 0.25
    ar_ratio_threshold: float = 3.0
    ar_max_position_shift_m: float = 0.02
    ar_expand_max_position_shift_m: float = 0.02
    ar_max_zwd_shift_m: float = 0.1
    ar_max_ambiguities: int = 12
    ar_constraint_sigma_m: float = 1.0e-4
    static_mode: bool = False
    static_position_noise_m_sqrt_s: float = 0.0
    static_zwd_noise_m_sqrt_s: float = 1.0e-4
    static_ambiguity_noise_m_sqrt_s: float = 0.0
    static_hold_min_fixed_epochs: int = 1100
    static_hold_window_epochs: int = 300
    static_hold_sigma_m: float = 0.01
    elevation_mask_deg: float = 10.0
    code_sigma_m: float = 0.45
    phase_sigma_m: float = 0.006
    doppler_sigma_mps: float = 0.12
    jerk_noise_mps3: float = 1.5
    isb_noise_m_sqrt_s: float = 0.01
    clock_sigma_m: float = 300.0
    clock_rate_noise_mps: float = 10.0
    zwd_noise_m_sqrt_s: float = 0.003
    ambiguity_noise_m_sqrt_s: float = 0.0003
    initial_position_sigma_m: float = 20.0
    initial_velocity_sigma_mps: float = 5.0
    initial_acceleration_sigma_mps2: float = 2.0
    initial_zwd_m: float = 0.12
    initial_zwd_sigma_m: float = 0.10
    max_epoch_gap_s: float = 120.0
    max_ambiguity_gap_s: float = 120.0
    gf_slip_threshold_m: float = 0.10
    mw_slip_threshold_m: float = 2.5
    prefit_gate_sigma: float = 8.0
    postfit_phase_gate_sigma: float = 6.0
    postfit_code_gate_sigma: float = 6.0
    spp_recovery_threshold_m: float = 80.0
    spp_recovery_min_failures: int = 5
    min_satellites: int = 4
    kinematic_min_satellites: int = 6
    time_direction: int = 1


@dataclass
class LooseCouplingMeasurement:
    week: int
    sow: float
    position: np.ndarray
    velocity: np.ndarray
    covariance_pv: np.ndarray
    status: int
    satellites: int
    phase_rms_m: float
    code_rms_m: float


@dataclass
class PPPSolution:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    clock_m: Dict[str, float]
    clock_rate_mps: float
    zwd_m: float
    covariance: np.ndarray
    status: int
    satellites: int
    measurements: int
    rejected_satellites: int = 0
    phase_rms_m: float = np.nan
    code_rms_m: float = np.nan
    doppler_rms_mps: float = np.nan
    fix_status: int = 0
    fixed_ambiguities: int = 0
    ar_ratio: float = 0.0
    model_status: Dict[str, str] = field(default_factory=dict)

    def loose_coupling(self, week: int, sow: float) -> LooseCouplingMeasurement:
        return LooseCouplingMeasurement(
            week, sow, self.position.copy(), self.velocity.copy(),
            self.covariance[:6, :6].copy(), self.status, self.satellites,
            self.phase_rms_m, self.code_rms_m,
        )


class PPPKalmanFilter:
    """Kinematic IF-float PPP with P/V covariance output for loose coupling.

    Base state: r(3), v(3), a(3), GPS clock, R/E/C/J ISB, clock rate,
    zenith wet delay, followed by dynamically allocated satellite ambiguities.
    """

    IDX_CLK = 9
    IDX_ISB = {"R": 10, "E": 11, "C": 12, "J": 13}
    IDX_CLK_RATE = 14
    IDX_ZWD = 15
    BASE_NX = 16

    def __init__(
        self,
        products: PreciseProducts,
        nav: Iterable[NavRecord] = (),
        config: Optional[PPPConfig] = None,
        antex: Optional[AntexCalibration] = None,
        erp: Optional[EarthRotationParameters] = None,
        receiver_antenna: str = "",
        osb: Optional[OsbCalibration] = None,
        obx: Optional[OrbexAttitude] = None,
        dcb: Optional[MonthlyCodeDcb] = None,
        antdel: Optional[np.ndarray] = None,
        ocean_loading: Optional[OceanLoading] = None,
    ):
        self.products, self.nav = products, list(nav)
        self.config = config or PPPConfig()
        self.antex, self.erp = antex, erp
        self.osb, self.obx, self.dcb = osb, obx, dcb
        self.receiver_antenna = receiver_antenna.strip()
        self.antdel = None if antdel is None else np.asarray(antdel, float).reshape(3)
        self.ocean_loading = ocean_loading
        self.spp = SPPSolver(elev_mask=self.config.elevation_mask_deg)
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.last_time: Optional[float] = None
        self.ambiguity_index: Dict[Tuple[str, int], int] = {}
        self.last_seen: Dict[Tuple[str, int], float] = {}
        self.geometry_free: Dict[Tuple[str, int], float] = {}
        self.melbourne_wubbena: Dict[Tuple[str, int], float] = {}
        self.phase_windup: Dict[Tuple[str, int], float] = {}
        self.mw_mean: Dict[Tuple[str, int], float] = {}
        self.mw_m2: Dict[Tuple[str, int], float] = {}
        self.mw_count: Dict[Tuple[str, int], int] = {}
        self.ambiguity_frequency: Dict[Tuple[str, int], Tuple[float, float]] = {}
        self.last_elevation: Dict[Tuple[str, int], float] = {}
        self.held_ambiguities: Dict[Tuple[Tuple[str, int], Tuple[str, int]], float] = {}
        self.static_fixed_positions = deque(maxlen=self.config.static_hold_min_fixed_epochs)
        self.static_position_anchor: Optional[np.ndarray] = None
        self.last_residuals: Dict[str, np.ndarray] = {}
        self.consecutive_failures = 0
        self.last_spp_position: Optional[np.ndarray] = None
        self.last_spp_time: Optional[float] = None
        self.smoothing_predicted_state: Optional[np.ndarray] = None
        self.smoothing_predicted_covariance: Optional[np.ndarray] = None
        self.smoothing_transition: Optional[np.ndarray] = None
        self.smoothing_transition_valid = False
        receiver_ok = bool(antex and antex.receiver(self.receiver_antenna) is not None)
        self.model_status = {
            "position_mode": "static" if self.config.static_mode else "kinematic",
            "precise_orbit_clock": "enabled",
            "relativity_sagnac_shapiro": "enabled",
            "satellite_antex": "enabled" if antex else "missing",
            "receiver_antex": "enabled" if receiver_ok else f"missing({self.receiver_antenna or 'unknown'})",
            "osb": "enabled" if osb else "missing",
            "obx_attitude": "enabled" if obx else "nominal-yaw",
            "code_dcb": "enabled" if dcb and (dcb.p1c1_m or dcb.p2c2_m) else "missing",
            "solid_pole_tide": "enabled" if erp else "solid-only(no ERP)",
            "ocean_loading": "enabled" if ocean_loading else "missing(BLQ)",
            "phase_windup": "enabled(obx)" if obx else "enabled(nominal-yaw)",
            "ambiguity": "PPP-AR(OSB)" if self.config.ambiguity_resolution and osb else ("float(OSB)" if osb else "float(no OSB/FCB)"),
        }

    @staticmethod
    def _frequency_pair(obs: SatelliteObservation):
        if obs.system in ("G", "J"):
            return 1575.42e6, 1227.60e6, f"{obs.system}01", f"{obs.system}02"
        if obs.system == "E":
            return 1575.42e6, 1176.45e6, "E01", "E05"
        if obs.system == "C":
            raw = obs.raw_observations or {}
            if obs.sat_id >= 19 and all(name in raw for name in ("C6I", "L6I")):
                return 1561.098e6, 1268.52e6, "C02", "C06"
            def band(code, default):
                c = (code or "").upper()
                table = {"1": (1575.42e6, "C01"), "2": (1561.098e6, "C02"),
                         "5": (1176.45e6, "C05"), "7": (1207.14e6, "C07"),
                         "6": (1268.52e6, "C06")}
                return table.get(c[1:2], default)
            q1 = band(obs.code_pr1, (1561.098e6, "C02"))
            q2 = band(obs.code_pr2, (1207.14e6, "C07"))
            return q1[0], q2[0], q1[1], q2[1]
        return None

    def _code_phase_bias(self, ob: SatelliteObservation, c1: str, c2: str) -> Tuple[float, float, float, float]:
        """Return metre biases for (P1, P2, L1, L2) after OSB/DCB lookup."""
        oc1 = OsbCalibration.obs_code("C", c1)
        oc2 = OsbCalibration.obs_code("C", c2)
        ol1 = OsbCalibration.obs_code("L", c1)
        ol2 = OsbCalibration.obs_code("L", c2)
        if self.osb:
            p1b = self.osb.code_bias(ob.system, ob.sat_id, oc1)
            p2b = self.osb.code_bias(ob.system, ob.sat_id, oc2)
            l1b = self.osb.phase_bias(ob.system, ob.sat_id, ol1)
            l2b = self.osb.phase_bias(ob.system, ob.sat_id, ol2)
        else:
            p1b = p2b = l1b = l2b = 0.0
        if self.dcb:
            p1b = self.dcb.supplement_code_bias(ob.system, ob.sat_id, oc1, p1b)
            p2b = self.dcb.supplement_code_bias(ob.system, ob.sat_id, oc2, p2b)
        return p1b, p2b, l1b, l2b

    def _combinations(self, obs: SatelliteObservation):
        """Build the product-consistent IF pair from the full RINEX record."""
        raw = obs.raw_observations or {}
        choices = {
            "G": (("1C",), ("2W", "2X", "2L"), 1575.42e6, 1227.60e6, "G01", "G02"),
            "R": (("1C", "1P"), ("2P",), 1575.42e6, 1246.00e6, "G01", "G02"),
            "E": (("1X", "1C"), ("5X", "5Q", "7I", "7X"), 1575.42e6, 1176.45e6, "E01", "E05"),
            "C": (("2I", "2X"), ("7I", "7X"), 1561.098e6, 1207.14e6, "C02", "C07"),
            "J": (("1C", "1X"), ("2X", "2L"), 1575.42e6, 1227.60e6, "J01", "J02"),
        }
        selected = None
        c1s = c2s = ""
        if obs.system in choices and raw:
            q1, q2, f1, f2, atx1, atx2 = choices[obs.system]
            if obs.system == "C" and obs.sat_id >= 19:
                q2, f2, atx2 = ("6I", "7Z", "7D"), 1268.52e6, "C06"
            for c1 in q1:
                for c2 in q2:
                    names = ("C"+c1, "C"+c2, "L"+c1, "L"+c2)
                    if all(n in raw and raw[n][0] not in (None, 0.0) for n in names):
                        selected = (raw[names[0]][0], raw[names[1]][0], raw[names[2]][0], raw[names[3]][0], raw[names[2]][1], raw[names[3]][1], raw.get("D"+c1, (None,0,0))[0], raw.get("D"+c2, (None,0,0))[0], f1, f2, atx1, atx2)
                        c1s, c2s = c1, c2
                        break
                if selected:
                    break
        if selected:
            p1,p2,lcy1,lcy2,lli1,lli2,d1,d2,f1,f2,atx1,atx2 = selected
        else:
            pair = self._frequency_pair(obs)
            vals = (obs.pseudorange_L1, obs.pseudorange_L2, obs.carrier_phase_L1, obs.carrier_phase_L2)
            if pair is None or any(v is None or not np.isfinite(v) or v == 0.0 for v in vals):
                return None
            p1,p2,lcy1,lcy2 = vals; lli1,lli2 = obs.lli_L1,obs.lli_L2
            d1,d2 = obs.doppler_L1,obs.doppler_L2; f1,f2,atx1,atx2 = pair
            defaults = {"G": ("1C", "2W"), "R": ("1C", "2P"), "E": ("1X", "7I"), "C": ("2I", "7I"), "J": ("1C", "2X")}
            c1s, c2s = defaults.get(obs.system, ("", ""))
        if not c1s or not c2s:
            return None
        den=f1*f1-f2*f2
        if abs(den)<1.0:
            return None
        p1b, p2b, l1b, l2b = self._code_phase_bias(obs, c1s, c2s)
        a,b=f1*f1/den,-f2*f2/den
        l1=float(lcy1)*CLIGHT/f1-l1b
        l2=float(lcy2)*CLIGHT/f2-l2b
        p1,p2=float(p1)-p1b,float(p2)-p2b
        phase,code=a*l1+b*l2,a*p1+b*p2
        gf=l1-l2
        mw=(f1*l1-f2*l2)/(f1-f2)-(f1*p1+f2*p2)/(f1+f2)
        doppler=None
        if d1 not in (None,0.0) and d2 not in (None,0.0):
            # RINEX Doppler is the negative carrier-phase range rate.
            doppler=-(a*float(d1)*CLIGHT/f1+b*float(d2)*CLIGHT/f2)
        return code,phase,gf,mw,doppler,f1,f2,atx1,atx2,a,b,int(lli1),int(lli2)

    def _initialize(self, epoch: GNSSRawObservation, approx: Optional[np.ndarray]):
        explicit = None if approx is None else np.asarray(approx, float).reshape(3)
        trust_approx = explicit is not None and np.all(np.isfinite(explicit)) and np.linalg.norm(explicit) >= 1e6
        seed = explicit if trust_approx else getattr(epoch, "approx_position", None)
        spp_pos, clk, ok = self.spp.solve(epoch, self.nav, seed) if self.nav else (seed, 0.0, int(seed is not None))
        p = explicit if trust_approx else spp_pos
        if not ok or p is None or np.linalg.norm(p) < 1e6: return False
        self.x = np.zeros(self.BASE_NX); self.x[:3] = p; self.x[self.IDX_CLK] = clk; self.x[self.IDX_ZWD] = self.config.initial_zwd_m
        sig = [self.config.initial_position_sigma_m]*3 + [self.config.initial_velocity_sigma_mps]*3 + [self.config.initial_acceleration_sigma_mps2]*3 + [self.config.clock_sigma_m] + [100.0]*4 + [50.0, self.config.initial_zwd_sigma_m]
        self.P = np.diag(np.square(sig)); self.last_time = epoch.week*604800.0+epoch.timestamp; self.last_spp_position = spp_pos.copy();self.last_spp_time=self.last_time
        if self.config.static_mode:
            self.x[3:9] = 0.0
            self.P[3:9, :] = 0.0
            self.P[:, 3:9] = 0.0
        self.smoothing_transition_valid = False
        return True

    def _add_ambiguity(self, key, value):
        assert self.x is not None and self.P is not None
        i = len(self.x); self.ambiguity_index[key] = i; self.x = np.append(self.x, value)
        q = np.zeros((i+1, i+1)); q[:i, :i] = self.P; q[i, i] = 100.0**2; self.P = q
        return i

    def _reset_ambiguity(self, key, value):
        i = self.ambiguity_index.get(key)
        if i is None: return self._add_ambiguity(key, value)
        self.x[i] = value; self.P[i, :] = 0.0; self.P[:, i] = 0.0; self.P[i, i] = 100.0**2
        return i

    def _rebase_held_ambiguities(self, slipped):
        if not self.held_ambiguities:
            return
        rebased={}
        for potentials in self._held_ambiguity_components():
            remaining=[key for key in potentials if key!=slipped]
            if len(remaining)<2:continue
            ref=max(remaining,key=lambda key:self.last_elevation.get(key,0.0))
            rebased.update({(key,ref):potentials[key]-potentials[ref] for key in remaining if key!=ref})
        self.held_ambiguities=rebased

    def _held_ambiguity_components(self):
        adjacency={}
        for (key,ref),target in self.held_ambiguities.items():
            adjacency.setdefault(key,[]).append((ref,target))
            adjacency.setdefault(ref,[]).append((key,-target))
        components=[];unvisited=set(adjacency)
        while unvisited:
            root=min(unvisited);potentials={root:0.0};stack=[root]
            while stack:
                key=stack.pop()
                for other,difference in sorted(adjacency.get(key,[]),key=lambda item:item[0]):
                    if other not in potentials:
                        potentials[other]=potentials[key]-difference;stack.append(other)
            unvisited.difference_update(potentials);components.append(potentials)
        return components

    def _predict(self, time):
        assert self.x is not None and self.P is not None and self.last_time is not None
        dt = time-self.last_time;h=abs(dt)
        if dt*self.config.time_direction <= 0.0 or h > self.config.max_epoch_gap_s: return False
        n = len(self.x); F = np.eye(n)
        Q = np.zeros((n, n))
        if self.config.static_mode:
            self.x[3:9] = 0.0
            self.P[3:9, :] = 0.0
            self.P[:, 3:9] = 0.0
            Q[:3, :3] = np.eye(3)*self.config.static_position_noise_m_sqrt_s**2*h
        else:
            F[:3, 3:6] = np.eye(3)*dt; F[:3, 6:9] = np.eye(3)*dt*dt/2; F[3:6, 6:9] = np.eye(3)*dt
            q = self.config.jerk_noise_mps3**2
            direction=1.0 if dt>0.0 else -1.0
            block = q*np.array([[h**5/20,direction*h**4/8,h**3/6],[direction*h**4/8,h**3/3,direction*h**2/2],[h**3/6,direction*h**2/2,h]])
            for k in range(3): Q[np.ix_([k,k+3,k+6],[k,k+3,k+6])] = block
        for i in self.IDX_ISB.values(): Q[i, i] = self.config.isb_noise_m_sqrt_s**2*h
        Q[self.IDX_CLK_RATE,self.IDX_CLK_RATE] = self.config.clock_rate_noise_mps**2*h
        zwd_noise = self.config.static_zwd_noise_m_sqrt_s if self.config.static_mode else self.config.zwd_noise_m_sqrt_s
        ambiguity_noise = self.config.static_ambiguity_noise_m_sqrt_s if self.config.static_mode else self.config.ambiguity_noise_m_sqrt_s
        Q[self.IDX_ZWD,self.IDX_ZWD] = zwd_noise**2*h
        if n > self.BASE_NX: Q[self.BASE_NX:,self.BASE_NX:] = np.eye(n-self.BASE_NX)*ambiguity_noise**2*h
        self.x = F@self.x; self.P = F@self.P@F.T+Q; self.last_time = time
        self.smoothing_predicted_state = self.x[:9].copy()
        self.smoothing_predicted_covariance = self.P[:9,:9].copy()
        self.smoothing_transition = F[:9,:9].copy()
        self.smoothing_transition_valid = True
        return True

    def _reset_clock(self, clock):
        i = self.IDX_CLK; self.x[i] = clock; self.P[i,:] = 0.0; self.P[:,i] = 0.0; self.P[i,i] = self.config.clock_sigma_m**2

    @staticmethod
    def _mapf(el, a, b, c):
        s = math.sin(el)
        return (1+a/(1+b/(1+c)))/(s+a/(s+b/(s+c)))

    @classmethod
    def _troposphere(cls, llh, el, gps_time):
        lat = abs(math.degrees(llh[0])); lats = np.array([15,30,45,60,75],float)
        ah=np.array([1.2769934,1.2683230,1.2465397,1.2196049,1.2045996])*1e-3; bh=np.array([2.9153695,2.9152299,2.9288445,2.9022565,2.9024912])*1e-3; ch=np.array([62.610505,62.837393,63.721774,63.824265,64.258455])*1e-3
        aa=np.array([0.0,1.2709626,2.6523662,3.4000452,4.1202191])*1e-5; ba=np.array([0.0,2.1414979,3.0160779,7.2562722,11.723375])*1e-5; ca=np.array([0.0,9.0128400,4.3497037,84.795348,170.37206])*1e-5
        aw=np.array([5.8021897,5.6794847,5.8118019,5.9727542,6.1641693])*1e-4; bw=np.array([1.4275268,1.5138625,1.4572752,1.5007428,1.7599082])*1e-3; cw=np.array([4.3472961,4.6729510,4.3908931,4.4626982,5.4736038])*1e-2
        interp=lambda x: float(np.interp(lat,lats,x)); dt=datetime(1980,1,6)+__import__('datetime').timedelta(seconds=gps_time)
        phase=2*math.pi*(dt.timetuple().tm_yday-(28 if llh[0]>=0 else 211))/365.25
        a,b,c=interp(ah)-interp(aa)*math.cos(phase),interp(bh)-interp(ba)*math.cos(phase),interp(ch)-interp(ca)*math.cos(phase)
        mh=cls._mapf(el,a,b,c); mw=cls._mapf(el,interp(aw),interp(bw),interp(cw)); h=max(-1000,min(10000,llh[2])); mh += (1/math.sin(el)-cls._mapf(el,2.53e-5,5.49e-3,1.14e-3))*h/1000
        pressure=1013.25*(1-2.2557e-5*h)**5.2568; zhd=0.0022768*pressure/(1-0.00266*math.cos(2*llh[0])-0.00028*h/1000)
        return zhd*mh,mw

    @staticmethod
    def _azel(sat, rec, llh):
        u=(sat-rec)/np.linalg.norm(sat-rec); sl,cl,sp,cp=math.sin(llh[1]),math.cos(llh[1]),math.sin(llh[0]),math.cos(llh[0])
        enu=np.array([[-sl,cl,0],[-sp*cl,-sp*sl,cp],[cp*cl,cp*sl,sp]])@u
        return math.atan2(enu[0],enu[1])%(2*math.pi),math.asin(np.clip(enu[2],-1,1))

    @staticmethod
    def _rotate(pos, vel, travel):
        a=OMGE*travel; c,s=math.cos(a),math.sin(a); R=np.array([[c,s,0],[-s,c,0],[0,0,1]])
        return R@pos,R@vel

    def _satellite(self, ob, time, code, f1, f2, atx1, atx2, sun):
        travel=code/CLIGHT
        for _ in range(2):
            full=self.products.state_full(ob.system,ob.sat_id,time-travel)
            if full is None: return None
            pos,vel,clk,drift=full; pos,vel=self._rotate(pos,vel,travel)
            if self.antex:
                pos += self.antex.satellite_if_offset(
                    ob.system, ob.sat_id, time, pos, sun, atx1, atx2, f1, f2, self.obx,
                )
            travel=np.linalg.norm(pos-self.x[:3])/CLIGHT
        return pos,vel,clk,drift

    def _build(self, epoch, time):
        assert self.x is not None and self.P is not None
        tide=earth_tide_displacement(self.x[:3],time,self.erp)
        if self.ocean_loading: tide += self.ocean_loading.displacement(self.x[:3],time,self.erp)
        rec=self.x[:3]+tide; llh=ecef2llh(rec); erpv=self.erp.at(time) if self.erp else np.zeros(4); sun,_,_=sun_moon_ecef(time,erpv[2])
        cand=[]
        for ob in epoch.observations:
            comb=self._combinations(ob)
            if comb is None: continue
            code,phase,gf,mw,dop,f1,f2,atx1,atx2,a,b,lli1,lli2=comb; sv=self._satellite(ob,time,code,f1,f2,atx1,atx2,sun)
            if sv is None: continue
            sat,svel,sclk,sdrift=sv
            sun_u=sun/np.linalg.norm(sun); shadow_axis=float(np.dot(sat,sun_u))
            if shadow_axis<0.0 and np.linalg.norm(sat-shadow_axis*sun_u)<6378137.0: continue
            az,el=self._azel(sat,rec,llh)
            if el < math.radians(self.config.elevation_mask_deg): continue
            key=(ob.system,ob.sat_id); slip=bool(ob.cycle_slip or lli1 or lli2)
            if key in self.geometry_free and abs(gf-self.geometry_free[key])>self.config.gf_slip_threshold_m: slip=True
            if key in self.melbourne_wubbena and abs(mw-self.melbourne_wubbena[key])>self.config.mw_slip_threshold_m: slip=True
            if key in self.last_seen and abs(time-self.last_seen[key])>self.config.max_ambiguity_gap_s: slip=True
            sat_axes = self.obx.body_axes(ob.system, ob.sat_id, time) if self.obx else None
            pcv=self.antex.satellite_if_pcv(ob.system,ob.sat_id,time,sat,rec,atx1,atx2,f1,f2) if self.antex else 0.0
            if self.antex:
                rec_pcv = self.antex.receiver_range_correction(
                    self.receiver_antenna, az, el, atx1, atx2, f1, f2, self.antdel,
                )
            elif self.antdel is not None:
                rec_pcv = AntexCalibration.receiver_antdel_correction(az, el, self.antdel)
            else:
                rec_pcv = 0.0
            phw=phase_windup_cycles(sat,rec,sun,self.phase_windup.get(key,0.0),sat_axes); self.phase_windup[key]=phw
            ant_corr=pcv+rec_pcv
            code-=ant_corr; phase-=ant_corr+phw*(a*CLIGHT/f1+b*CLIGHT/f2)
            if key not in self.ambiguity_index or slip:
                if slip:
                    self._rebase_held_ambiguities(key)
                self._reset_ambiguity(key,phase-code)
                self.mw_mean.pop(key,None);self.mw_m2.pop(key,None);self.mw_count.pop(key,None)
            frequency=(f1,f2)
            if self.ambiguity_frequency.get(key) not in (None,frequency):
                self.mw_mean.pop(key,None);self.mw_m2.pop(key,None);self.mw_count.pop(key,None)
            self.ambiguity_frequency[key]=frequency;count=self.mw_count.get(key,0)+1;mean=self.mw_mean.get(key,0.0);delta=mw-mean;mean+=delta/count
            self.mw_count[key]=count;self.mw_mean[key]=mean;self.mw_m2[key]=self.mw_m2.get(key,0.0)+delta*(mw-mean);self.last_elevation[key]=el
            self.geometry_free[key],self.melbourne_wubbena[key],self.last_seen[key]=gf,mw,time
            cand.append((ob,key,code,phase,dop,sat,svel,sclk,sdrift,el))
        H=[]; v=[]; var=[]; kinds=[]; keys=[]
        for ob,key,code,phase,dop,sat,svel,sclk,sdrift,el in cand:
            diff=rec-sat; rho=np.linalg.norm(diff); e=diff/rho; hydro,mwmap=self._troposphere(llh,el,time)
            shapiro=2*GM_EARTH/CLIGHT**2*math.log((np.linalg.norm(sat)+np.linalg.norm(rec)+rho)/(np.linalg.norm(sat)+np.linalg.norm(rec)-rho))
            clk=self.x[self.IDX_CLK]+(self.x[self.IDX_ISB[ob.system]] if ob.system in self.IDX_ISB else 0.0)
            common=rho+shapiro+clk-CLIGHT*sclk+hydro+mwmap*self.x[self.IDX_ZWD]
            h=np.zeros(len(self.x)); h[:3]=e; h[self.IDX_CLK]=1; h[self.IDX_ZWD]=mwmap
            if ob.system in self.IDX_ISB: h[self.IDX_ISB[ob.system]]=1
            sinel=max(math.sin(el),0.15); pair=self._frequency_pair(ob); f1,f2=pair[:2]; den=f1*f1-f2*f2; aa,bb=f1*f1/den,-f2*f2/den
            cv=(aa*aa+bb*bb)*(self.config.code_sigma_m/sinel)**2; lv=(aa*aa+bb*bb)*(self.config.phase_sigma_m/sinel)**2
            H.append(h);v.append(code-common);var.append(cv);kinds.append('C');keys.append(key)
            hp=h.copy();hp[self.ambiguity_index[key]]=1;H.append(hp);v.append(phase-common-self.x[self.ambiguity_index[key]]);var.append(lv);kinds.append('L');keys.append(key)
            if dop is not None and not self.config.static_mode:
                model=float(np.dot(e,self.x[3:6]-svel))+self.x[self.IDX_CLK_RATE]-CLIGHT*sdrift
                hd=np.zeros(len(self.x));hd[3:6]=e;hd[self.IDX_CLK_RATE]=1
                H.append(hd);v.append(dop-model);var.append((self.config.doppler_sigma_mps/sinel)**2);kinds.append('D');keys.append(key)
        return cand,np.asarray(H),np.asarray(v),np.asarray(var),np.asarray(kinds),keys

    def _robust_update(self,H,v,var,kinds,keys):
        if len(v)==0: return np.zeros(0,bool),np.zeros(0)
        x0,P0=self.x.copy(),self.P.copy(); mask=np.ones(len(v),bool)
        Sdiag=np.einsum('ij,jk,ik->i',H,P0,H)+var; norm=np.abs(v)/np.sqrt(np.maximum(Sdiag,1e-12))
        bad={keys[i] for i in range(len(v)) if not np.isfinite(norm[i]) or norm[i]>self.config.prefit_gate_sigma}; mask=np.array([k not in bad for k in keys])
        for _ in range(3):
            required=self.config.min_satellites if self.config.static_mode else self.config.kinematic_min_satellites
            if len(set(k for i,k in enumerate(keys) if mask[i]))<required: return np.zeros(len(v),bool),np.full(len(v),np.nan)
            self.x,self.P=x0.copy(),P0.copy(); HH,vv,R=H[mask],v[mask],np.diag(var[mask]); S=HH@self.P@HH.T+R
            try: K=np.linalg.solve(S,HH@self.P).T
            except np.linalg.LinAlgError: return np.zeros(len(v),bool),np.full(len(v),np.nan)
            dx=K@vv; self.x+=dx; A=np.eye(len(self.x))-K@HH; self.P=A@self.P@A.T+K@R@K.T; self.P=(self.P+self.P.T)/2
            post=np.full(len(v),np.nan);post[mask]=vv-HH@dx; score=np.zeros(len(v)); score[mask]=np.abs(post[mask])/np.sqrt(var[mask])
            offenders=[]
            for i in np.where(mask)[0]:
                lim=self.config.postfit_phase_gate_sigma if kinds[i]=='L' else self.config.postfit_code_gate_sigma
                if kinds[i]!='D' and score[i]>lim: offenders.append(i)
            if not offenders: return mask,post
            worst=max(offenders,key=lambda i:score[i]); badkey=keys[worst];mask &= np.array([k!=badkey for k in keys])
        return mask,post

    def apply_navigation_prior(self,position,position_cov,velocity=None,velocity_cov=None):
        if self.x is None: raise RuntimeError('PPP filter is not initialized')
        idx=list(range(3)); z=list(np.asarray(position)-self.x[:3]); vv=list(np.diag(position_cov) if np.asarray(position_cov).ndim==2 else position_cov)
        if velocity is not None:
            idx+=list(range(3,6));z+=list(np.asarray(velocity)-self.x[3:6]);vv+=list(np.diag(velocity_cov) if np.asarray(velocity_cov).ndim==2 else velocity_cov)
        H=np.zeros((len(idx),len(self.x)));H[np.arange(len(idx)),idx]=1;R=np.diag(vv);S=H@self.P@H.T+R;K=np.linalg.solve(S,H@self.P).T;dx=K@np.asarray(z);self.x+=dx;A=np.eye(len(self.x))-K@H;self.P=A@self.P@A.T+K@R@K.T

    def _apply_static_position_hold(self, fix_status):
        if not self.config.static_mode or not self.config.ambiguity_resolution:
            return
        if fix_status and self.static_position_anchor is None:
            self.static_fixed_positions.append(self.x[:3].copy())
            if len(self.static_fixed_positions)>=self.config.static_hold_min_fixed_epochs:
                positions=np.asarray(self.static_fixed_positions)
                window=positions[-self.config.static_hold_window_epochs:]
                self.static_position_anchor=np.median(window,axis=0)
        if self.static_position_anchor is None:
            return
        H=np.zeros((3,len(self.x)));H[:,:3]=np.eye(3)
        R=np.eye(3)*self.config.static_hold_sigma_m**2
        innovation=self.static_position_anchor-H@self.x;S=H@self.P@H.T+R
        try:K=np.linalg.solve(S,H@self.P).T
        except np.linalg.LinAlgError:return
        self.x+=K@innovation;A=np.eye(len(self.x))-K@H
        self.P=A@self.P@A.T+K@R@K.T;self.P=(self.P+self.P.T)/2

    def _try_fix_ambiguities(self):
        if not self.config.ambiguity_resolution or self.osb is None or self.x is None or self.P is None:
            return 0,0,0.0
        active=[];components=self._held_ambiguity_components();held_nodes=set()
        for potentials in components:
            held_nodes.update(potentials)
            visible=sorted(key for key in potentials if key in self.ambiguity_index and self.last_seen.get(key)==self.last_time)
            if visible:
                ref=max(visible,key=lambda key:self.last_elevation.get(key,0.0))
                for key in visible:
                    if key==ref:continue
                    row=np.zeros(len(self.x));row[self.ambiguity_index[key]]=1.0;row[self.ambiguity_index[ref]]=-1.0
                    active.append((row,potentials[key]-potentials[ref]))
        held_result=None
        if len(active)>=self.config.ar_min_satellites-1:
            H=np.vstack([item[0] for item in active]);target=np.asarray([item[1] for item in active]);R=np.eye(len(H))*self.config.ar_constraint_sigma_m**2;innovation=target-H@self.x;S=H@self.P@H.T+R
            try:K=np.linalg.solve(S,H@self.P).T
            except np.linalg.LinAlgError:self.held_ambiguities.clear()
            else:
                x_float=self.x.copy();P_float=self.P.copy();self.x+=K@innovation;A=np.eye(len(self.x))-K@H;self.P=A@self.P@A.T+K@R@K.T;self.P=(self.P+self.P.T)/2
                if np.all(np.isfinite(self.x)) and np.linalg.norm(self.x[:3]-x_float[:3])<=self.config.ar_max_position_shift_m and abs(self.x[self.IDX_ZWD]-x_float[self.IDX_ZWD])<=self.config.ar_max_zwd_shift_m:
                    held_result=(1,len(active),float('inf'))
                else:
                    self.x,self.P=x_float,P_float;self.held_ambiguities.clear();held_nodes.clear()
        groups={}
        for key,index in self.ambiguity_index.items():
            count=self.mw_count.get(key,0);freq=self.ambiguity_frequency.get(key)
            if count<self.config.ar_min_epochs or freq is None or self.last_elevation.get(key,0.0)<math.radians(self.config.ar_min_elevation_deg):continue
            wl=CLIGHT/(freq[0]-freq[1]);m2=self.mw_m2.get(key,0.0);sigma=math.sqrt(max(m2/(count-1),0.0)/count)/abs(wl) if count>1 else np.inf
            if sigma>self.config.ar_wide_lane_sigma_cycles:continue
            groups.setdefault((key[0],round(freq[0]),round(freq[1])),[]).append(key)
        candidates=[]
        for group,keys in sorted(groups.items()):
            keys=sorted(keys)
            if len(keys)<self.config.ar_min_satellites:continue
            ref=max(keys,key=lambda k:self.last_elevation.get(k,0.0));f1,f2=self.ambiguity_frequency[ref];wl=CLIGHT/(f1-f2);lam1=CLIGHT/f1;lamnl=CLIGHT/(f1+f2);alpha=f1*f1/(f1*f1-f2*f2)
            ref_wl=self.mw_mean[ref]/wl
            for key in keys:
                if key==ref:continue
                dwl_float=self.mw_mean[key]/wl-ref_wl;dwl=int(round(dwl_float))
                if abs(dwl_float-dwl)>self.config.ar_wide_lane_fraction:continue
                row=np.zeros(len(self.x));row[self.ambiguity_index[key]]=1.0;row[self.ambiguity_index[ref]]=-1.0
                nl_float=(float(row@self.x)-alpha*lam1*dwl)/lamnl;nl_sigma=math.sqrt(max(float(row@self.P@row),0.0))/lamnl
                if nl_sigma>self.config.ar_narrow_lane_sigma_cycles:continue
                candidates.append((nl_sigma,nl_float,row,alpha*lam1*dwl,lamnl,key,ref))
        if len(candidates)<self.config.ar_min_satellites-1:return held_result or (0,0,0.0)
        if held_result and all(item[5] in held_nodes and item[6] in held_nodes for item in candidates):
            return held_result
        candidates.sort(key=lambda item:item[0]);candidates=candidates[:self.config.ar_max_ambiguities]
        afloat=np.asarray([item[1] for item in candidates]);T=np.vstack([item[2]/item[4] for item in candidates]);Q=T@self.P@T.T;Q=(Q+Q.T)/2+np.eye(len(Q))*1e-10
        fixed,scores,ok=lambda_algorithm(afloat,Q,2)
        ratio=0.0 if not ok or len(scores)<2 or not np.all(np.isfinite(scores)) else (float('inf') if scores[0]<=1e-12 else float(scores[1]/scores[0]))
        if ratio>=self.config.ar_ratio_threshold:
            integers=np.rint(fixed[:,0])
        else:
            candidates=[item for item in candidates if abs(item[1]-round(item[1]))<self.config.ar_partial_narrow_lane_fraction and self.last_elevation.get(item[5],0.0)>=math.radians(self.config.ar_partial_min_elevation_deg) and self.last_elevation.get(item[6],0.0)>=math.radians(self.config.ar_partial_min_elevation_deg)]
            if len(candidates)<self.config.ar_min_satellites-1:return held_result or (0,0,ratio)
            integers=np.rint([item[1] for item in candidates])
        H=np.vstack([item[2] for item in candidates]);target=np.asarray([item[3]+item[4]*integers[i] for i,item in enumerate(candidates)])
        R=np.eye(len(H))*self.config.ar_constraint_sigma_m**2;innovation=target-H@self.x;S=H@self.P@H.T+R
        try:K=np.linalg.solve(S,H@self.P).T
        except np.linalg.LinAlgError:return held_result or (0,0,ratio)
        x_float=self.x.copy();P_float=self.P.copy();self.x+=K@innovation;A=np.eye(len(self.x))-K@H;self.P=A@self.P@A.T+K@R@K.T;self.P=(self.P+self.P.T)/2
        position_shift=np.linalg.norm(self.x[:3]-x_float[:3]);zwd_shift=abs(self.x[self.IDX_ZWD]-x_float[self.IDX_ZWD])
        position_limit=self.config.ar_expand_max_position_shift_m if held_result else self.config.ar_max_position_shift_m
        if not np.all(np.isfinite(self.x)) or position_shift>position_limit or zwd_shift>self.config.ar_max_zwd_shift_m:
            self.x,self.P=x_float,P_float
            return held_result or (0,0,ratio)
        self.held_ambiguities.update({(item[5],item[6]):float(target[i]) for i,item in enumerate(candidates)})
        return 1,len(candidates),ratio

    def process(self,epoch:GNSSRawObservation,approx_position:Optional[np.ndarray]=None)->PPPSolution:
        time=epoch.week*604800.0+epoch.timestamp
        if self.x is None:
            if not self._initialize(epoch,approx_position): return self._empty()
        elif time!=self.last_time and not self._predict(time):
            seed=self.x[:3].copy();self.x=self.P=None;self.ambiguity_index.clear();self.last_seen.clear();self.geometry_free.clear();self.melbourne_wubbena.clear();self.mw_mean.clear();self.mw_m2.clear();self.mw_count.clear();self.ambiguity_frequency.clear();self.last_elevation.clear();self.held_ambiguities.clear()
            if not self._initialize(epoch,seed): return self._empty()
        spp_pos,spp_clk,spp_ok=self.spp.solve(epoch,self.nav,self.x[:3]) if self.nav else (self.x[:3],self.x[self.IDX_CLK],1)
        if spp_ok:
            self._reset_clock(spp_clk)
            explicit_approx = approx_position is not None and np.all(np.isfinite(approx_position)) and np.linalg.norm(approx_position)>=1e6
            spp_innovation=np.linalg.norm(self.x[:3]-spp_pos)
            recover_from_spp=(
                self.consecutive_failures>=self.config.spp_recovery_min_failures
                and len(epoch.observations)>=self.config.kinematic_min_satellites
                and spp_innovation<=self.config.spp_recovery_threshold_m
            )
            if not self.config.static_mode and not explicit_approx and recover_from_spp:
                spp_dt=time-self.last_spp_time if self.last_spp_time is not None else float(self.config.time_direction);prev=self.last_spp_position;self.x[:3]=spp_pos
                if prev is not None and abs(spp_dt)>1e-6:self.x[3:6]=(spp_pos-prev)/spp_dt
                self.P[:9,:]=0;self.P[:,:9]=0;self.P[:3,:3]=np.eye(3)*25;self.P[3:6,3:6]=np.eye(3)*25;self.P[6:9,6:9]=np.eye(3)*4
                self.smoothing_transition_valid=False
            self.last_spp_position=spp_pos.copy();self.last_spp_time=time
        cand,H,v,var,kinds,keys=self._build(epoch,time)
        required=self.config.min_satellites if self.config.static_mode else self.config.kinematic_min_satellites
        mask,post=self._robust_update(H,v,var,kinds,keys) if len(cand)>=required else (np.zeros(len(v),bool),np.full(len(v),np.nan))
        used=set(k for i,k in enumerate(keys) if i<len(mask) and mask[i]);status=int(len(used)>=required)
        self.consecutive_failures=0 if status else self.consecutive_failures+1
        fix_status,fixed_ambiguities,ar_ratio=self._try_fix_ambiguities() if status else (0,0,0.0)
        if status:self._apply_static_position_hold(fix_status)
        if not -0.05<self.x[self.IDX_ZWD]<0.8:self.x[self.IDX_ZWD]=self.config.initial_zwd_m;self.P[self.IDX_ZWD,self.IDX_ZWD]=self.config.initial_zwd_sigma_m**2
        self.last_residuals={k:post[(kinds==k)&mask] for k in ('C','L','D')} if len(post) else {}
        rms=lambda a:float(np.sqrt(np.mean(a*a))) if len(a) else np.nan;clocks={'G':float(self.x[self.IDX_CLK])}
        for s,i in self.IDX_ISB.items():clocks[s]=float(self.x[self.IDX_CLK]+self.x[i])
        cov=self.P[:9,:9].copy();return PPPSolution(self.x[:3].copy(),self.x[3:6].copy(),self.x[6:9].copy(),clocks,float(self.x[self.IDX_CLK_RATE]),float(self.x[self.IDX_ZWD]),cov,status,len(used),int(np.count_nonzero(mask)),max(0,len(set(keys))-len(used)),rms(self.last_residuals.get('L',[])),rms(self.last_residuals.get('C',[])),rms(self.last_residuals.get('D',[])),fix_status,fixed_ambiguities,ar_ratio,self.model_status.copy())

    def _empty(self):
        return PPPSolution(np.full(3,np.nan),np.full(3,np.nan),np.full(3,np.nan),{},np.nan,np.nan,np.empty((0,0)),0,0,0,model_status=self.model_status.copy())
