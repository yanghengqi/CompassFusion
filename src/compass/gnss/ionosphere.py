"""
电离层延迟模型

实现Klobuchar模型（GPS广播电离层模型）和BDS电离层模型
参考: COMPASS src/LibGnss/rtkcmn.c klobuchar_GPS() & klobuchar_BDS()
"""

import numpy as np
from typing import Tuple, Optional
from ..core.constants import CLIGHT

# 默认GPS电离层参数 (2004/1/1)
ION_DEFAULT_GPS = np.array([
    0.1118e-07, -0.7451e-08, -0.5961e-07, 0.1192e-06,  # alpha0-3
    0.1167e+06, -0.2294e+06, -0.1311e+06, 0.1049e+07   # beta0-3
])

# 地球半径 (km)
RE_IONO = 6378.0
# 电离层高度 (km)
HION = 350.0


def klobuchar_gps(gps_time: float, ion: np.ndarray, pos: np.ndarray, 
                  azel: np.ndarray) -> float:
    """
    GPS Klobuchar电离层延迟模型
    
    计算L1频率的电离层延迟（单位：米）
    
    参考: COMPASS klobuchar_GPS() in rtkcmn.c
    
    Args:
        gps_time: GPS周内秒 (s)
        ion: 电离层参数 [a0,a1,a2,a3,b0,b1,b2,b3]
             a0-a3: alpha参数 (s, s/semi-circle, s/semi-circle^2, s/semi-circle^3)
             b0-b3: beta参数 (s, s/semi-circle, s/semi-circle^2, s/semi-circle^3)
        pos: 接收机位置 [lat, lon, h] (rad, rad, m)
        azel: 方位角和高度角 [az, el] (rad)
    
    Returns:
        电离层延迟 L1 (m)
    """
    # 参数检查
    if pos[2] < -1e3 or azel[1] <= 0.0:
        return 0.0
    
    # 使用默认参数
    if ion is None or np.linalg.norm(ion) <= 0.0:
        ion = ION_DEFAULT_GPS
    
    # 地心角 (semi-circle)
    psi = 0.0137 / (azel[1] / np.pi + 0.11) - 0.022
    
    # 亚电离层点纬度/经度 (semi-circle)
    phi = pos[0] / np.pi + psi * np.cos(azel[0])
    if phi > 0.416:
        phi = 0.416
    elif phi < -0.416:
        phi = -0.416
    
    lam = pos[1] / np.pi + psi * np.sin(azel[0]) / np.cos(phi * np.pi)
    
    # 地磁纬度 (semi-circle)
    phi += 0.064 * np.cos((lam - 1.617) * np.pi)
    
    # 本地时间 (s)
    tt = 43200.0 * lam + gps_time
    tt -= np.floor(tt / 86400.0) * 86400.0  # 0 <= tt < 86400
    
    # 倾斜因子
    f = 1.0 + 16.0 * pow(0.53 - azel[1] / np.pi, 3.0)
    
    # 电离层延迟
    amp = ion[0] + phi * (ion[1] + phi * (ion[2] + phi * ion[3]))
    per = ion[4] + phi * (ion[5] + phi * (ion[6] + phi * ion[7]))
    
    amp = max(amp, 0.0)
    per = max(per, 72000.0)
    
    x = 2.0 * np.pi * (tt - 50400.0) / per
    
    if abs(x) < 1.57:
        delay = 5e-9 + amp * (1.0 + x * x * (-0.5 + x * x / 24.0))
    else:
        delay = 5e-9
    
    return CLIGHT * f * delay


def ionppp(pos: np.ndarray, azel: np.ndarray, re: float, hion: float) -> Tuple[float, np.ndarray]:
    """
    计算电离层穿刺点位置和倾斜因子
    
    参考: COMPASS ionppp() in rtkcmn.c
    
    Args:
        pos: 接收机位置 [lat, lon, h] (rad, rad, m)
        azel: 方位角和高度角 [az, el] (rad)
        re: 地球半径 (km)
        hion: 电离层高度 (km)
    
    Returns:
        (slant_factor, posp): 倾斜因子和穿刺点位置 [lat, lon, h] (rad)
    """
    rp = re / (re + hion) * np.cos(azel[1])
    ap = np.pi / 2.0 - azel[1] - np.arcsin(rp)
    sinap = np.sin(ap)
    tanap = np.tan(ap)
    cosaz = np.cos(azel[0])
    
    posp = np.zeros(3)
    posp[0] = np.arcsin(np.sin(pos[0]) * np.cos(ap) + np.cos(pos[0]) * sinap * cosaz)
    
    if (pos[0] > 70.0 * np.pi / 180.0 and tanap * cosaz > np.tan(np.pi / 2.0 - pos[0])) or \
       (pos[0] < -70.0 * np.pi / 180.0 and -tanap * cosaz > np.tan(np.pi / 2.0 + pos[0])):
        posp[1] = pos[1] + np.pi - np.arcsin(sinap * np.sin(azel[0]) / np.cos(posp[0]))
    else:
        posp[1] = pos[1] + np.arcsin(sinap * np.sin(azel[0]) / np.cos(posp[0]))
    
    # 倾斜因子
    slant_factor = 1.0 / np.sqrt(1.0 - rp * rp)
    
    return slant_factor, posp


def klobuchar_bds(gps_time: float, ion: np.ndarray, pos: np.ndarray,
                  azel: np.ndarray) -> float:
    """
    BDS Klobuchar电离层延迟模型
    
    参考: COMPASS klobuchar_BDS() in rtkcmn.c
    
    Args:
        gps_time: GPS周内秒 (s)
        ion: BDS电离层参数 [a0,a1,a2,a3,b0,b1,b2,b3]
        pos: 接收机位置 [lat, lon, h] (rad, rad, m)
        azel: 方位角和高度角 [az, el] (rad)
    
    Returns:
        电离层延迟 B1 (m)
    """
    # 参数检查
    if pos[2] < -1e3 or azel[1] <= 0.0:
        return 0.0
    
    # 如果没有BDS电离层参数，使用GPS模型
    if ion is None or np.linalg.norm(ion) <= 0.0:
        return klobuchar_gps(gps_time, None, pos, azel)
    
    # BDS电离层模型参数
    re = 6378.0  # 地球半径 (km)
    hion = 375.0  # 电离层高度 (km)
    
    # 计算穿刺点
    f, blhp = ionppp(pos, azel, re, hion)
    
    # 本地时间
    dt = 43200.0 * blhp[1] / np.pi + gps_time
    dt -= np.floor(dt / 86400.0) * 86400.0
    
    # 地磁纬度
    phi = blhp[0] / np.pi
    
    # 计算延迟
    amp = ion[0] + phi * (ion[1] + phi * (ion[2] + phi * ion[3]))
    per = ion[4] + phi * (ion[5] + phi * (ion[6] + phi * ion[7]))
    
    amp = max(amp, 0.0)
    per = max(per, 72000.0)
    per = min(per, 172800.0)
    
    x = dt - 50400.0
    
    if abs(x) < per / 4.0:
        delay = 5e-9 + amp * np.cos(2.0 * np.pi * x / per)
    else:
        delay = 5e-9
    
    return CLIGHT * f * delay


def ionmapf(pos: np.ndarray, azel: np.ndarray) -> float:
    """
    电离层映射函数（单层模型）
    
    参考: COMPASS ionmapf() in rtkcmn.c
    
    Args:
        pos: 接收机位置 [lat, lon, h] (rad, rad, m)
        azel: 方位角和高度角 [az, el] (rad)
    
    Returns:
        电离层映射函数值
    """
    HION_M = 350000.0  # 电离层高度 (m)
    RE_WGS84 = 6378137.0  # WGS84地球半径 (m)
    
    if pos[2] >= HION_M:
        return 1.0
    
    return 1.0 / np.cos(np.arcsin((RE_WGS84 + pos[2]) / (RE_WGS84 + HION_M) * np.sin(np.pi / 2.0 - azel[1])))


def ionocorr(gps_time: float, ion_gps: Optional[np.ndarray], ion_bds: Optional[np.ndarray],
             pos: np.ndarray, azel: np.ndarray, system: str = 'G') -> Tuple[float, float]:
    """
    计算电离层延迟改正和方差
    
    Args:
        gps_time: GPS周内秒 (s)
        ion_gps: GPS电离层参数 [a0-a3, b0-b3]，可为None使用默认值
        ion_bds: BDS电离层参数 [a0-a3, b0-b3]，可为None
        pos: 接收机位置 [lat, lon, h] (rad, rad, m)
        azel: 方位角和高度角 [az, el] (rad)
        system: 卫星系统 ('G'=GPS, 'C'=BDS, 'E'=Galileo, 'R'=GLONASS)
    
    Returns:
        (dion, vion): 电离层延迟 (m) 和方差 (m^2)
    """
    ERR_BRDCI = 0.5  # 广播电离层模型误差因子
    
    if system == 'C':
        # BDS使用自己的电离层模型
        dion = klobuchar_bds(gps_time, ion_bds, pos, azel)
    else:
        # GPS/Galileo/GLONASS/QZSS使用GPS电离层模型
        dion = klobuchar_gps(gps_time, ion_gps, pos, azel)
    
    # 电离层延迟方差
    vion = (dion * ERR_BRDCI) ** 2
    
    return dion, vion


def freq_dependent_iono(dion_L1: float, freq_L1: float, freq_target: float) -> float:
    """
    计算不同频率的电离层延迟
    
    电离层延迟与频率平方成反比: dion_f = dion_L1 * (f_L1/f)^2
    
    Args:
        dion_L1: L1频率的电离层延迟 (m)
        freq_L1: L1频率 (Hz)
        freq_target: 目标频率 (Hz)
    
    Returns:
        目标频率的电离层延迟 (m)
    """
    return dion_L1 * (freq_L1 / freq_target) ** 2
