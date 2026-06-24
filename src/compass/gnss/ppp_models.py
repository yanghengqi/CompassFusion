"""Physical correction models used by the float PPP estimator.

The implementations follow the model layout used by RTKLIB/COMPASS: ERP is
interpolated, satellite antenna offsets are formed in the yaw-steering body
frame, solid/pole tides displace the receiver, and phase wind-up is kept
continuous in cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from .orbex_attitude import OrbexAttitude
import math
import numpy as np

from ..core.constants import CLIGHT, wgs84
from ..core.transforms import ecef2llh

AS2R = np.pi / (180.0 * 3600.0)
GPS_EPOCH = datetime(1980, 1, 6)
GME, GMS, GMM = 3.986004415e14, 1.327124e20, 4.902801e12


@dataclass
class OceanLoading:
    coefficients: np.ndarray

    @classmethod
    def from_file(cls, path: str, station: str) -> "OceanLoading":
        target = station.strip().upper()
        with open(path, "r", encoding="ascii", errors="replace") as stream:
            lines = iter(stream)
            for line in lines:
                if line.startswith("$$") or not line.strip():
                    continue
                if line.split()[0].upper() != target:
                    continue
                rows = []
                for record in lines:
                    if record.startswith("$$") or not record.strip():
                        continue
                    values = [float(value) for value in record.split()]
                    if len(values) == 11:
                        rows.append(values)
                    if len(rows) == 6:
                        return cls(np.asarray(rows, float))
                break
        raise ValueError(f"no BLQ ocean-loading coefficients for {target}")

    def displacement(self, receiver: np.ndarray, gps_seconds: float, erp: Optional["EarthRotationParameters"] = None) -> np.ndarray:
        args = np.array([
            [1.40519e-4, 2.0, -2.0, 0.0, 0.00],
            [1.45444e-4, 0.0, 0.0, 0.0, 0.00],
            [1.37880e-4, 2.0, -3.0, 1.0, 0.00],
            [1.45842e-4, 2.0, 0.0, 0.0, 0.00],
            [0.72921e-4, 1.0, 0.0, 0.0, 0.25],
            [0.67598e-4, 1.0, -2.0, 0.0, -0.25],
            [0.72523e-4, -1.0, 0.0, 0.0, -0.25],
            [0.64959e-4, 1.0, -3.0, 1.0, -0.25],
            [0.53234e-5, 0.0, 2.0, 0.0, 0.00],
            [0.26392e-5, 0.0, 1.0, -1.0, 0.00],
            [0.03982e-5, 2.0, 0.0, 0.0, 0.00],
        ])
        ut1_utc = erp.at(gps_seconds)[2] if erp else 0.0
        dt = _datetime_from_gpst(gps_seconds) + timedelta(seconds=float(ut1_utc))
        midnight = datetime(dt.year, dt.month, dt.day)
        fday = (dt - midnight).total_seconds()
        days = (midnight - datetime(1975, 1, 1)).total_seconds() / 86400.0 + 1.0
        t = (27392.500528 + 1.000000035 * days) / 36525.0
        a = np.array([
            fday,
            math.radians(279.69668 + 36000.768930485*t + 3.03e-4*t*t),
            math.radians(270.434358 + 481267.88314137*t - 0.001133*t*t + 1.9e-6*t**3),
            math.radians(334.329653 + 4069.0340329577*t - 0.010325*t*t - 1.2e-5*t**3),
            2.0 * math.pi,
        ])
        angle = args @ a
        phase = np.radians(self.coefficients[3:6])
        dp = np.sum(self.coefficients[:3] * np.cos(angle[None, :] - phase), axis=1)
        enu = np.array([-dp[1], -dp[2], dp[0]])
        lat, lon = ecef2llh(receiver)[:2]
        sl, cl, sp, cp = math.sin(lon), math.cos(lon), math.sin(lat), math.cos(lat)
        return np.array([
            -sl*enu[0] - sp*cl*enu[1] + cp*cl*enu[2],
            cl*enu[0] - sp*sl*enu[1] + cp*sl*enu[2],
            cp*enu[1] + sp*enu[2],
        ])


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return None if n <= 0.0 else np.asarray(v, float) / n


def _datetime_from_gpst(gps_seconds: float) -> datetime:
    # Valid for the supplied 2021 products. Leap-second table can replace this
    # constant when pre-2017 data is introduced.
    return GPS_EPOCH + timedelta(seconds=float(gps_seconds) - 18.0)


def _julian_day(dt: datetime) -> float:
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + (dt.minute + (dt.second + dt.microsecond * 1e-6) / 60.0) / 60.0) / 24.0
    if m <= 2:
        y -= 1; m += 12
    a = y // 100
    b = 2 - a + a // 4
    return math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5


def _gmst(jd: float, ut1_utc: float = 0.0) -> float:
    jd += ut1_utc / 86400.0
    t = (jd - 2451545.0) / 36525.0
    deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t - t ** 3 / 38710000.0
    return math.radians(deg % 360.0)


def _eci_to_ecef(v: np.ndarray, gmst: float) -> np.ndarray:
    c, s = math.cos(gmst), math.sin(gmst)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])


def sun_moon_ecef(gps_seconds: float, ut1_utc: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Low-precision Sun/Moon ECEF coordinates adequate for tide/yaw models."""
    dt = _datetime_from_gpst(gps_seconds)
    jd = _julian_day(dt)
    d = jd - 2451543.5
    eps = math.radians(23.4393 - 3.563e-7 * d)

    # Sun (Schlyter low-precision analytical orbit).
    w = math.radians((282.9404 + 4.70935e-5 * d) % 360.0)
    e = 0.016709 - 1.151e-9 * d
    M = math.radians((356.0470 + 0.9856002585 * d) % 360.0)
    E = M + e * math.sin(M) * (1.0 + e * math.cos(M))
    xv, yv = math.cos(E) - e, math.sqrt(1.0 - e * e) * math.sin(E)
    lon = math.atan2(yv, xv) + w
    r_au = math.hypot(xv, yv)
    au = 149597870700.0
    sun_eci = au * r_au * np.array([math.cos(lon), math.sin(lon) * math.cos(eps), math.sin(lon) * math.sin(eps)])

    # Moon geocentric orbit, sufficient for centimetre-level solid tide model.
    N = math.radians((125.1228 - 0.0529538083 * d) % 360.0)
    inc = math.radians(5.1454)
    wm = math.radians((318.0634 + 0.1643573223 * d) % 360.0)
    Mm = math.radians((115.3654 + 13.0649929509 * d) % 360.0)
    em, am = 0.054900, 60.2666 * 6378137.0
    Em = Mm
    for _ in range(5):
        Em -= (Em - em * math.sin(Em) - Mm) / (1.0 - em * math.cos(Em))
    x, y = am * (math.cos(Em) - em), am * math.sqrt(1 - em * em) * math.sin(Em)
    rm, vm = math.hypot(x, y), math.atan2(y, x)
    lonm = vm + wm
    xe = rm * (math.cos(N) * math.cos(lonm) - math.sin(N) * math.sin(lonm) * math.cos(inc))
    ye = rm * (math.sin(N) * math.cos(lonm) + math.cos(N) * math.sin(lonm) * math.cos(inc))
    ze = rm * math.sin(lonm) * math.sin(inc)
    moon_eci = np.array([xe, ye * math.cos(eps) - ze * math.sin(eps), ye * math.sin(eps) + ze * math.cos(eps)])
    gst = _gmst(jd, ut1_utc)
    return _eci_to_ecef(sun_eci, gst), _eci_to_ecef(moon_eci, gst), gst


@dataclass
class EarthRotationParameters:
    rows: np.ndarray

    @classmethod
    def from_file(cls, path: str) -> "EarthRotationParameters":
        rows = []
        with open(path, "r", encoding="ascii", errors="ignore") as stream:
            for line in stream:
                p = line.split()
                if len(p) < 5:
                    continue
                try:
                    mjd = float(p[0]); xp = float(p[1]) * 1e-6 * AS2R; yp = float(p[2]) * 1e-6 * AS2R
                    ut1 = float(p[3]) * 1e-7; lod = float(p[4]) * 1e-7
                except ValueError:
                    continue
                if 40000.0 < mjd < 100000.0:
                    rows.append((mjd, xp, yp, ut1, lod))
        if not rows:
            raise ValueError(f"No ERP records in {path}")
        return cls(np.asarray(rows, float))

    def at(self, gps_seconds: float) -> np.ndarray:
        dt = _datetime_from_gpst(gps_seconds)
        mjd = _julian_day(dt) - 2400000.5
        t = self.rows[:, 0]
        j = int(np.searchsorted(t, mjd))
        if j <= 0: return self.rows[0, 1:].copy()
        if j >= len(t): return self.rows[-1, 1:].copy()
        f = (mjd - t[j - 1]) / (t[j] - t[j - 1])
        return (1 - f) * self.rows[j - 1, 1:] + f * self.rows[j, 1:]


@dataclass
class AntennaFrequency:
    pco: np.ndarray
    pcv: np.ndarray
    zen1: float
    dzen: float


@dataclass
class AntennaCalibration:
    frequencies: Dict[str, AntennaFrequency]


@dataclass
class AntexCalibration:
    satellites: Dict[str, list[Tuple[float, float, AntennaCalibration]]] = field(default_factory=dict)
    receivers: Dict[str, AntennaCalibration] = field(default_factory=dict)
    receiver_types: set[str] = field(default_factory=set)

    @classmethod
    def from_file(cls, path: str) -> "AntexCalibration":
        obj = cls(); block = None; ant_type = ""; freq = None; fdata = {}; valid_from = -np.inf; valid_until = np.inf; zen1 = 0.0; dzen = 1.0
        with open(path, "r", encoding="ascii", errors="ignore") as stream:
            for line in stream:
                label = line[60:].strip() if len(line) >= 60 else ""
                if label == "START OF ANTENNA":
                    block = None; ant_type = ""; freq = None; fdata = {}; valid_from = -np.inf; valid_until = np.inf
                elif label == "TYPE / SERIAL NO":
                    ant_type, serial = line[:20].strip(), line[20:40].strip()
                    block = serial if serial and serial[0:1] in "GRECIJ" else None
                    if block is None and ant_type:
                        obj.receiver_types.add(ant_type)
                elif label == "ZEN1 / ZEN2 / DZEN":
                    try: zen1, _zen2, dzen = map(float, line[:60].split()[:3])
                    except ValueError: pass
                elif label in ("VALID FROM", "VALID UNTIL"):
                    try:
                        p = line[:43].split(); sec = float(p[5]); dt = datetime(*map(int, p[:5]), int(sec))
                        value = (dt - GPS_EPOCH).total_seconds() + 18.0
                        if label == "VALID FROM": valid_from = value
                        else: valid_until = value
                    except (ValueError, TypeError): pass
                elif label == "START OF FREQUENCY":
                    freq = line[3:6].strip(); fdata[freq] = {"pco": np.zeros(3), "pcv": np.zeros(0)}
                elif label == "NORTH / EAST / UP" and freq:
                    try: fdata[freq]["pco"] = np.asarray(list(map(float, line[:60].split()[:3]))) * 1e-3
                    except ValueError: pass
                elif line.startswith("   NOAZI") and freq:
                    try: fdata[freq]["pcv"] = np.asarray(list(map(float, line[8:].split()))) * 1e-3
                    except ValueError: pass
                elif label == "END OF ANTENNA":
                    cal = AntennaCalibration({k: AntennaFrequency(v["pco"], v["pcv"], zen1, dzen) for k, v in fdata.items()})
                    if block:
                        obj.satellites.setdefault(block, []).append((valid_from, valid_until, cal))
                    elif ant_type:
                        obj.receivers[ant_type] = cal
        return obj

    def receiver(self, ant_type: str) -> Optional[AntennaCalibration]:
        ant_type = (ant_type or "").strip()
        if not ant_type:
            return None
        if ant_type in self.receivers:
            return self.receivers[ant_type]
        upper = ant_type.upper()
        for name, cal in self.receivers.items():
            if name.upper() == upper:
                return cal
        for name, cal in self.receivers.items():
            if upper in name.upper() or name.upper() in upper:
                return cal
        return None

    def satellite(self, system: str, sat_id: int, time: float) -> Optional[AntennaCalibration]:
        for start, end, cal in self.satellites.get(f"{system}{sat_id:02d}", []):
            if start <= time <= end: return cal
        return None

    @staticmethod
    def _body_axes(sat: np.ndarray, sun: np.ndarray):
        ez = _unit(-sat); es = _unit(sun - sat)
        if ez is None or es is None: return None
        ey = _unit(np.cross(ez, es))
        if ey is None: return None
        ex = np.cross(ey, ez)
        return ex, ey, ez

    def satellite_if_offset(
        self,
        system: str,
        sat_id: int,
        time: float,
        sat: np.ndarray,
        sun: np.ndarray,
        f1_code: str,
        f2_code: str,
        f1: float,
        f2: float,
        attitude: Optional["OrbexAttitude"] = None,
    ) -> np.ndarray:
        cal = self.satellite(system, sat_id, time)
        axes = attitude.body_axes(system, sat_id, time) if attitude is not None else None
        if axes is None:
            axes = self._body_axes(sat, sun)
        if cal is None or axes is None:
            return np.zeros(3)
        a, b = f1 * f1 / (f1 * f1 - f2 * f2), -f2 * f2 / (f1 * f1 - f2 * f2)
        ex, ey, ez = axes
        def vec(code):
            af = cal.frequencies.get(code)
            return np.zeros(3) if af is None else af.pco[0] * ex + af.pco[1] * ey + af.pco[2] * ez
        return a * vec(f1_code) + b * vec(f2_code)

    @staticmethod
    def receiver_antdel_correction(az: float, el: float, antdel: np.ndarray) -> float:
        """Project ENU antenna delta onto line-of-sight (used when receiver ATX is missing)."""
        del_enu = np.asarray(antdel, float).reshape(3)
        if not np.any(del_enu):
            return 0.0
        cosel = math.cos(el)
        los = np.array([math.sin(az) * cosel, math.cos(az) * cosel, math.sin(el)])
        return -float(np.dot(del_enu, los))

    def receiver_range_correction(
        self,
        ant_type: str,
        az: float,
        el: float,
        f1_code: str,
        f2_code: str,
        f1: float,
        f2: float,
        antdel: Optional[np.ndarray] = None,
    ) -> float:
        """RTKLIB-style receiver PCO+PCV projected on the line-of-sight (IF combination)."""
        del_enu = np.zeros(3) if antdel is None else np.asarray(antdel, float).reshape(3)
        cal = self.receiver(ant_type)
        if cal is None:
            return self.receiver_antdel_correction(az, el, del_enu)
        cosel = math.cos(el)
        los = np.array([math.sin(az) * cosel, math.cos(az) * cosel, math.sin(el)])
        zen = 90.0 - math.degrees(el)
        den = f1 * f1 - f2 * f2
        if abs(den) < 1.0:
            return 0.0
        a, b = f1 * f1 / den, -f2 * f2 / den

        def one(code: str, weight: float) -> float:
            af = cal.frequencies.get(code)
            if af is None:
                return 0.0
            off = af.pco + del_enu
            pco = -float(np.dot(off, los))
            if len(af.pcv) == 0:
                return weight * pco
            x = (zen - af.zen1) / af.dzen
            pcv = float(np.interp(x, np.arange(len(af.pcv)), af.pcv))
            return weight * (pco + pcv)

        return one(f1_code, a) + one(f2_code, b)

    def satellite_if_pcv(self, system: str, sat_id: int, time: float, sat: np.ndarray, receiver: np.ndarray, f1_code: str, f2_code: str, f1: float, f2: float) -> float:
        cal = self.satellite(system, sat_id, time)
        if cal is None: return 0.0
        nadir = math.degrees(math.acos(np.clip(np.dot(_unit(-sat), _unit(receiver-sat)), -1.0, 1.0)))
        a, b = f1*f1/(f1*f1-f2*f2), -f2*f2/(f1*f1-f2*f2)
        def pcv(code):
            af = cal.frequencies.get(code)
            if af is None or len(af.pcv) == 0: return 0.0
            x = (nadir - af.zen1) / af.dzen
            return float(np.interp(x, np.arange(len(af.pcv)), af.pcv))
        return a * pcv(f1_code) + b * pcv(f2_code)


def phase_windup_cycles(
    sat: np.ndarray,
    receiver: np.ndarray,
    sun: np.ndarray,
    previous: float,
    sat_body_axes: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> float:
    axes = sat_body_axes if sat_body_axes is not None else AntexCalibration._body_axes(sat, sun)
    ek = _unit(receiver - sat)
    if axes is None or ek is None:
        return previous
    exs, eys, _ = axes
    llh = ecef2llh(receiver); lat, lon = llh[:2]
    north = np.array([-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)])
    west = np.array([math.sin(lon), -math.cos(lon), 0.0])
    ds = exs - ek * np.dot(ek, exs) - np.cross(ek, eys)
    dr = north - ek * np.dot(ek, north) + np.cross(ek, west)
    c = np.clip(np.dot(ds, dr) / (np.linalg.norm(ds) * np.linalg.norm(dr)), -1.0, 1.0)
    ph = math.acos(c) / (2.0 * math.pi)
    if np.dot(ek, np.cross(ds, dr)) < 0.0: ph = -ph
    return ph + math.floor(previous - ph + 0.5)


def _tide_planet(up: np.ndarray, body: np.ndarray, gm: float, llh: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(body); ep = body / r; a = float(np.dot(ep, up))
    p = (3.0 * math.sin(llh[0])**2 - 1.0) / 2.0
    h2, l2, h3, l3 = 0.6078 - 0.0006*p, 0.0847 + 0.0002*p, 0.292, 0.015
    k2 = gm/GME * wgs84.a**4/r**3; k3 = k2*wgs84.a/r
    dp = k2*3*l2*a + k3*l3*(7.5*a*a-1.5)
    du = k2*(h2*(1.5*a*a-0.5)-3*l2*a*a) + k3*(h3*(2.5*a**3-1.5*a)-l3*(7.5*a*a-1.5)*a)
    return dp*ep + du*up


def earth_tide_displacement(receiver: np.ndarray, gps_seconds: float, erp: Optional[EarthRotationParameters]) -> np.ndarray:
    erpv = erp.at(gps_seconds) if erp else np.zeros(4)
    sun, moon, gst = sun_moon_ecef(gps_seconds, erpv[2])
    llh = ecef2llh(receiver); up = _unit(receiver)
    dr = _tide_planet(up, sun, GMS, llh) + _tide_planet(up, moon, GMM, llh)
    dr += -0.012 * math.sin(2*llh[0]) * math.sin(gst+llh[1]) * up
    if erp:
        dt = _datetime_from_gpst(gps_seconds); y = (dt-datetime(2000,1,1)).total_seconds()/86400/365.25
        xp_bar, yp_bar = 23.513+7.6141*y, 358.891-0.6287*y
        m1, m2 = erpv[0]/AS2R-xp_bar*1e-3, -erpv[1]/AS2R+yp_bar*1e-3
        e = np.array([-math.sin(llh[1]), math.cos(llh[1]), 0.0])
        n = np.array([-math.sin(llh[0])*math.cos(llh[1]), -math.sin(llh[0])*math.sin(llh[1]), math.cos(llh[0])])
        de = 9e-3*math.sin(llh[0])*(m1*math.sin(llh[1])-m2*math.cos(llh[1]))
        dn = -9e-3*math.cos(2*llh[0])*(m1*math.cos(llh[1])+m2*math.sin(llh[1]))
        du = -33e-3*math.sin(2*llh[0])*(m1*math.cos(llh[1])+m2*math.sin(llh[1]))
        dr += de*e + dn*n + du*up
    return dr
