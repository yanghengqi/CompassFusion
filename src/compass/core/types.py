"""
核心数据类型定义
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


# 基础向量和矩阵类型
Vector3 = np.ndarray  # shape (3,)
Matrix3x3 = np.ndarray  # shape (3, 3)


@dataclass
class Quaternion:
    """四元数类 [q0, q1, q2, q3]，其中q0为标量部分"""
    w: float  # 标量部分
    x: float  # 向量部分i
    y: float  # 向量部分j
    z: float  # 向量部分k
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组 [w, x, y, z]"""
        return np.array([self.w, self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Quaternion':
        """从numpy数组创建四元数"""
        return cls(w=arr[0], x=arr[1], y=arr[2], z=arr[3])
    
    def normalize(self) -> 'Quaternion':
        """归一化四元数"""
        arr = self.to_array()
        norm = np.linalg.norm(arr)
        arr = arr / norm
        return Quaternion.from_array(arr)
    
    def conjugate(self) -> 'Quaternion':
        """共轭四元数"""
        return Quaternion(w=self.w, x=-self.x, y=-self.y, z=-self.z)
    
    def multiply(self, other: 'Quaternion') -> 'Quaternion':
        """四元数乘法 self ⊗ other"""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w=w, x=x, y=y, z=z)


@dataclass
class IMUData:
    """IMU数据"""
    timestamp: float  # GPS时间（秒）
    gyro: Vector3  # 陀螺仪角增量 [rad]
    accel: Vector3  # 加速度计速度增量 [m/s]
    dt: float  # 时间间隔 [s]
    
    @property
    def angular_rate(self) -> Vector3:
        """角速率 [rad/s]"""
        return self.gyro / self.dt if self.dt > 0 else self.gyro
    
    @property
    def specific_force(self) -> Vector3:
        """比力 [m/s²]"""
        return self.accel / self.dt if self.dt > 0 else self.accel


@dataclass
class GNSSData:
    """GNSS观测数据"""
    timestamp: float  # GPS时间（秒）
    position: Vector3  # 位置 [lat(rad), lon(rad), height(m)]
    velocity: Vector3  # 速度 [m/s]，NED坐标系
    position_std: Vector3  # 位置标准差 [m]
    velocity_std: Vector3  # 速度标准差 [m/s]
    num_satellites: int  # 卫星数量
    fix_type: int  # 定位类型: 0-无效, 1-单点, 4-固定解, 5-浮点解


@dataclass
class INSSolution:
    """INS解算结果"""
    timestamp: float  # GPS时间（秒）
    
    # 姿态
    quaternion: Quaternion  # 四元数姿态（b系到n系）
    euler: Vector3  # 欧拉角 [roll, pitch, yaw] (rad)
    
    # 速度
    velocity: Vector3  # NED坐标系速度 [m/s]
    
    # 位置
    position: Vector3  # [lat(rad), lon(rad), height(m)]
    
    # IMU零偏估计
    gyro_bias: Vector3  # 陀螺仪零偏 [rad/s]
    accel_bias: Vector3  # 加速度计零偏 [m/s²]
    
    # 协方差（对角元素）
    position_std: Vector3  # 位置标准差 [m]
    velocity_std: Vector3  # 速度标准差 [m/s]
    attitude_std: Vector3  # 姿态标准差 [rad]
    
    # 状态标志
    status: int  # 解算状态


@dataclass
class IMUConfig:
    """IMU配置参数"""
    # 噪声参数
    gyro_noise: float = 0.0001  # 陀螺仪角度随机游走 [rad/s]
    accel_noise: float = 0.001  # 加速度计速度随机游走 [m/s²]
    gyro_bias_rw: float = 0.00001  # 陀螺仪零偏随机游走 [rad/s²]
    accel_bias_rw: float = 0.0001  # 加速度计零偏随机游走 [m/s³]
    
    # 零偏马尔科夫过程相关时间
    gyro_bias_corr_time: float = 3600.0  # 陀螺仪零偏相关时间 [s]
    accel_bias_corr_time: float = 3600.0  # 加速度计零偏相关时间 [s]
    
    # 初始零偏
    init_gyro_bias: Vector3 = None  # 初始陀螺仪零偏 [rad/s]
    init_accel_bias: Vector3 = None  # 初始加速度计零偏 [m/s²]
    
    # 刻度因子（可选）
    gyro_scale: Vector3 = None  # 陀螺仪刻度因子误差
    accel_scale: Vector3 = None  # 加速度计刻度因子误差
    
    # 安装参数（可选）
    body_to_imu_dcm: Matrix3x3 = None  # IMU到载体坐标系的旋转矩阵
    lever_arm: Vector3 = None  # 杆臂 [m]，IMU到GNSS天线
    
    def __post_init__(self):
        """初始化默认值"""
        if self.init_gyro_bias is None:
            self.init_gyro_bias = np.zeros(3)
        if self.init_accel_bias is None:
            self.init_accel_bias = np.zeros(3)
        if self.gyro_scale is None:
            self.gyro_scale = np.zeros(3)
        if self.accel_scale is None:
            self.accel_scale = np.zeros(3)
        if self.body_to_imu_dcm is None:
            self.body_to_imu_dcm = np.eye(3)
        if self.lever_arm is None:
            self.lever_arm = np.zeros(3)


@dataclass
class GNSSConfig:
    """GNSS配置参数"""
    mode: str = 'PPK'  # 定位模式: 'SPP', 'PPK', 'PPP'
    
    # 观测噪声
    position_std: float = 0.02  # 位置测量标准差 [m]（固定解）
    velocity_std: float = 0.01  # 速度测量标准差 [m/s]
    
    # PPK参数
    baseline_length_max: float = 50000.0  # 最大基线长度 [m]
    min_satellites: int = 4  # 最小卫星数
    elevation_mask: float = 15.0  # 高度角截止角 [degree]
    
    # 系统选择
    use_gps: bool = True
    use_glonass: bool = False
    use_galileo: bool = False
    use_bds: bool = True
    
    # 频率选择
    use_l1: bool = True
    use_l2: bool = True
    
    # 电离层模型
    iono_model: str = 'BRDC'  # 'OFF', 'BRDC', 'IONO-FREE'
    
    # 对流层模型
    trop_model: str = 'SAAS'  # 'OFF', 'SAAS', 'EST'


@dataclass  
class ESKFState:
    """ESKF状态向量"""
    # 误差状态（15维核心状态）
    position_err: Vector3 = None  # 位置误差 [m]
    velocity_err: Vector3 = None  # 速度误差 [m/s]
    attitude_err: Vector3 = None  # 姿态误差（小角度）[rad]
    accel_bias_err: Vector3 = None  # 加速度计零偏误差 [m/s²]
    gyro_bias_err: Vector3 = None  # 陀螺仪零偏误差 [rad/s]
    
    # 状态协方差矩阵
    covariance: np.ndarray = None  # P矩阵
    
    def __post_init__(self):
        """初始化默认值"""
        if self.position_err is None:
            self.position_err = np.zeros(3)
        if self.velocity_err is None:
            self.velocity_err = np.zeros(3)
        if self.attitude_err is None:
            self.attitude_err = np.zeros(3)
        if self.accel_bias_err is None:
            self.accel_bias_err = np.zeros(3)
        if self.gyro_bias_err is None:
            self.gyro_bias_err = np.zeros(3)
        if self.covariance is None:
            self.covariance = np.eye(15) * 1e-6
    
    def to_vector(self) -> np.ndarray:
        """转换为向量形式"""
        return np.concatenate([
            self.position_err,
            self.velocity_err,
            self.attitude_err,
            self.accel_bias_err,
            self.gyro_bias_err
        ])
    
    def from_vector(self, vec: np.ndarray):
        """从向量更新状态"""
        self.position_err = vec[0:3]
        self.velocity_err = vec[3:6]
        self.attitude_err = vec[6:9]
        self.accel_bias_err = vec[9:12]
        self.gyro_bias_err = vec[12:15]
