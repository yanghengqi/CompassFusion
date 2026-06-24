"""
GNSS原始观测数据结构
支持紧组合和半紧组合
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from .types import Vector3


@dataclass
class SatelliteObservation:
    """单颗卫星的原始观测数据"""
    sat_id: int  # 卫星编号（PRN）
    system: str  # 卫星系统: 'G'-GPS, 'C'-BDS, 'E'-Galileo, 'R'-GLONASS
    
    # 伪距观测 [m]
    pseudorange_L1: float  # L1频点伪距
    pseudorange_L2: Optional[float] = None  # L2频点伪距
    
    # 载波相位观测 [cycle]
    carrier_phase_L1: Optional[float] = None  # L1载波相位
    carrier_phase_L2: Optional[float] = None  # L2载波相位
    
    # 多普勒观测 [Hz]
    doppler_L1: Optional[float] = None
    doppler_L2: Optional[float] = None
    
    # 信噪比 [dB-Hz]
    snr_L1: Optional[float] = None
    snr_L2: Optional[float] = None
    
    # 失锁指示符
    lli_L1: int = 0  # Loss of Lock Indicator
    lli_L2: int = 0
    
    # 卫星位置和钟差（来自星历）
    sat_position: Optional[Vector3] = None  # ECEF [m]
    sat_velocity: Optional[Vector3] = None  # ECEF [m/s]
    sat_clock_bias: Optional[float] = None  # 卫星钟差 [s]
    sat_clock_drift: Optional[float] = None  # 卫星钟漂 [s/s]
    
    # 高度角和方位角（用户位置计算）
    elevation: Optional[float] = None  # 高度角 [rad]
    azimuth: Optional[float] = None  # 方位角 [rad]
    
    # 观测质量
    cycle_slip: bool = False  # 周跳标志
    outlier: bool = False  # 粗差标志

    # RINEX 选用的伪距码类型（用于 prange/TGD 与 IF 组合对齐 pntpos.c）
    code_pr1: Optional[str] = None
    code_pr2: Optional[str] = None
    # 本历元该星所有 C* 伪距原始值（码类型 -> m），便于诊断
    obs_codes_c: Optional[Dict[str, float]] = None
    # Complete RINEX observation map retained for PPP signal selection.
    raw_observations: Optional[Dict[str, Tuple[float, int, float]]] = None


@dataclass
class GNSSRawObservation:
    """GNSS原始观测数据（一个历元）"""
    timestamp: float  # GPS时间 [s]
    week: int  # GPS周
    
    # 观测数据列表
    observations: List[SatelliteObservation]
    
    # RINEX 头 APPROX POSITION XYZ 时由读取器填入，为 ECEF [m]（非 LLH）
    approx_position: Optional[Vector3] = None
    
    @property
    def num_satellites(self) -> int:
        """卫星数量"""
        return len(self.observations)
    
    def get_gps_satellites(self) -> List[SatelliteObservation]:
        """获取GPS卫星观测"""
        return [obs for obs in self.observations if obs.system == 'G']
    
    def get_bds_satellites(self) -> List[SatelliteObservation]:
        """获取BDS卫星观测"""
        return [obs for obs in self.observations if obs.system == 'C']
    
    def get_satellite_by_id(self, sat_id: int, system: str) -> Optional[SatelliteObservation]:
        """根据卫星ID查找观测"""
        for obs in self.observations:
            if obs.sat_id == sat_id and obs.system == system:
                return obs
        return None


@dataclass
class NavigationData:
    """导航星历数据"""
    sat_id: int
    system: str
    epoch: float  # GPS时间 [s]
    
    # 广播星历参数（GPS/BDS）
    toe: float  # 星历参考时间
    toc: float  # 钟差参考时间
    
    # 卫星钟差参数
    af0: float  # 钟差 [s]
    af1: float  # 钟漂 [s/s]
    af2: float  # 钟漂率 [s/s²]
    
    # 轨道参数
    sqrt_a: float  # 轨道长半轴平方根 [m^0.5]
    e: float  # 偏心率
    i0: float  # 轨道倾角 [rad]
    omega0: float  # 升交点赤经 [rad]
    omega: float  # 近地点幅角 [rad]
    M0: float  # 平近点角 [rad]
    
    # 轨道摄动
    delta_n: float  # 平均角速度改正 [rad/s]
    omega_dot: float  # 升交点赤经变化率 [rad/s]
    i_dot: float  # 轨道倾角变化率 [rad/s]
    cuc: float  # 纬度幅角余弦改正 [rad]
    cus: float  # 纬度幅角正弦改正 [rad]
    crc: float  # 轨道半径余弦改正 [m]
    crs: float  # 轨道半径正弦改正 [m]
    cic: float  # 轨道倾角余弦改正 [rad]
    cis: float  # 轨道倾角正弦改正 [rad]
    
    # 群延迟
    tgd: float = 0.0   # TGD (GPS/QZSS) / BGD_E5aE1 (GAL) / TGD1 (BDS) [s]
    tgd2: float = 0.0  # (可选) BGD_E5bE1 (GAL) / TGD2 (BDS) [s]
    
    # 健康状态
    health: int = 0
    ura: float = 2.0  # 用户距离精度 [m]
    
    def is_valid(self, time: float, max_age: float = 7200.0) -> bool:
        """检查星历是否有效"""
        if self.health != 0:
            return False
        if abs(time - self.toe) > max_age:
            return False
        return True


@dataclass
class GlonassEphemeris:
    """GLONASS 广播星历（对齐 RTKLIB rinex.c decode_geph / geph_t）。"""

    sat_id: int
    system: str = "GLONASS"
    toe: float = 0.0  # 参考时刻 [s]，与导航读取器一致（GPST 连续秒）
    toc: float = 0.0
    taun: float = 0.0  # 存 -τn（RINEX 第一钟差项取负），与 RTKLIB geph->taun 一致
    gamn: float = 0.0  # γn（频间钟差斜率项）[s/s]
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # PZ-90 ECEF [m]
    vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    acc: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    svh: int = 0
    frq: int = 0  # 频率号 k（-7..13）
    age: int = 0

    def is_valid(self, time: float, max_age: float = 7200.0) -> bool:
        if self.svh != 0:
            return False
        if abs(time - self.toe) > max_age:
            return False
        return True


NavRecord = Union[NavigationData, GlonassEphemeris]


@dataclass  
class PreciseEphemeris:
    """精密星历数据（SP3格式）"""
    timestamp: float  # GPS时间 [s]
    
    # 卫星位置和钟差
    satellite_positions: dict  # {(system, sat_id): position_xyz}
    satellite_clocks: dict  # {(system, sat_id): clock_bias}
    
    def get_satellite_state(self, sat_id: int, system: str) -> tuple:
        """获取卫星位置和钟差"""
        key = (system, sat_id)
        pos = self.satellite_positions.get(key)
        clk = self.satellite_clocks.get(key)
        return pos, clk
