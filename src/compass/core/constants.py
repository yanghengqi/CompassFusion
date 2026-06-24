"""
物理和数学常量
"""

import numpy as np
from dataclasses import dataclass


# 数学常量
PI = np.pi
D2R = PI / 180.0  # 度转弧度
R2D = 180.0 / PI  # 弧度转度

# 物理常量
CLIGHT = 299792458.0  # 光速 [m/s]
OMGE_GPS = 7.2921151467e-5  # GPS地球自转角速度 [rad/s]


# WGS84椭球参数
@dataclass
class WGS84:
    """WGS84地球椭球参数"""
    a: float = 6378137.0  # 长半轴 [m]
    f: float = 1.0 / 298.257223563  # 扁率
    e: float = 0.0818191908426  # 第一偏心率
    e2: float = 0.00669437999013  # 第一偏心率平方
    omega_ie: float = 7.2921151467e-5  # 地球自转角速度 [rad/s]
    GM: float = 3.986004418e14  # 地球引力常数 [m³/s²]
    
    @property
    def b(self) -> float:
        """短半轴 [m]"""
        return self.a * (1 - self.f)
    
    @property
    def R0(self) -> float:
        """平均地球半径 [m]"""
        return (2 * self.a + self.b) / 3


# 全局WGS84实例
wgs84 = WGS84()


@dataclass
class EarthParams:
    """地球参数（局部）"""
    RMh: float  # 子午圈曲率半径 + 高度 [m]
    RNh: float  # 卯酉圈曲率半径 + 高度 [m]
    latitude: float  # 纬度 [rad]
    height: float  # 高度 [m]
    
    @property
    def omega_ie(self) -> np.ndarray:
        """地球自转角速度在n系的投影"""
        return np.array([
            wgs84.omega_ie * np.cos(self.latitude),
            0.0,
            -wgs84.omega_ie * np.sin(self.latitude)
        ])
    
    @property
    def omega_en(self) -> np.ndarray:
        """牵连角速度（导航系相对地球系）"""
        # 在NED坐标系下
        return np.array([0.0, 0.0, 0.0])  # 简化，实际需要根据速度计算
    
    @property
    def gravity(self) -> np.ndarray:
        """重力加速度在n系的投影 [m/s²]"""
        # Somigliana重力公式
        sin_lat = np.sin(self.latitude)
        sin_lat2 = sin_lat ** 2
        
        # 正常重力（椭球面上）
        g0 = 9.7803253359 * (1 + 0.001931853 * sin_lat2) / \
             np.sqrt(1 - wgs84.e2 * sin_lat2)
        
        # 高度修正
        g = g0 * (1 - 2 * self.height / wgs84.a)
        
        # NED坐标系，重力沿D（Down）方向
        return np.array([0.0, 0.0, g])
    
    @classmethod
    def from_position(cls, lat: float, lon: float, height: float) -> 'EarthParams':
        """
        从位置计算地球参数
        
        Args:
            lat: 纬度 [rad]
            lon: 经度 [rad]
            height: 高度 [m]
        
        Returns:
            EarthParams实例
        """
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lat2 = sin_lat ** 2
        
        # 子午圈曲率半径
        RM = wgs84.a * (1 - wgs84.e2) / (1 - wgs84.e2 * sin_lat2) ** 1.5
        
        # 卯酉圈曲率半径
        RN = wgs84.a / np.sqrt(1 - wgs84.e2 * sin_lat2)
        
        return cls(
            RMh=RM + height,
            RNh=RN + height,
            latitude=lat,
            height=height
        )


# 物理常量
CLIGHT = 299792458.0  # 光速 [m/s]
G = 9.7803267715  # 平均重力加速度 [m/s²]
G2MPS2 = G  # 重力常量


# 单位转换
UG2MPS2 = G / 1e6  # ug -> m/s²
MG2MPS2 = G / 1e3  # mg -> m/s²
DPH2RPS = D2R / 3600.0  # deg/hour -> rad/s
DPS2RPS = D2R  # deg/s -> rad/s


# 坐标系枚举
class CoordSystem:
    """坐标系枚举"""
    ECEF = 0  # 地心地固坐标系
    LLH = 1  # 大地坐标系
    ENU = 2  # 东北天坐标系
    NED = 3  # 北东地坐标系


class IMUCoord:
    """IMU坐标系枚举"""
    FRD = 0  # Forward-Right-Down（前右下）
    RFU = 1  # Right-Forward-Up（右前上）


# 解算状态枚举
class SolutionStatus:
    """解算状态"""
    NONE = 0  # 无解
    FIX = 1  # 固定解
    FLOAT = 2  # 浮点解
    SBAS = 3  # SBAS
    DGPS = 4  # DGPS/DGNSS
    SINGLE = 5  # 单点定位
    PPP = 6  # PPP
    DR = 7  # 航位推算
    
    # INS状态
    INS_MECH = 10  # 纯INS机械编排
    INS_ALIGN = 11  # INS对准中
    LC = 12  # 松组合
    TC = 13  # 紧组合


# 初始化协方差
class InitCovariance:
    """初始化协方差标准差"""
    POSITION_H = 10.0  # 水平位置 [m]
    POSITION_V = 10.0  # 垂直位置 [m]
    VELOCITY = 1.0  # 速度 [m/s]
    ATTITUDE_RP = 1.0 * D2R  # 横滚俯仰 [rad]
    ATTITUDE_YAW = 5.0 * D2R  # 航向 [rad]
    GYRO_BIAS = 100.0 * DPH2RPS  # 陀螺仪零偏 [rad/s]
    ACCEL_BIAS = 1000.0 * UG2MPS2  # 加速度计零偏 [m/s²]
