"""
坐标转换和旋转相关函数
"""

import numpy as np
from typing import Tuple
from .types import Quaternion, Vector3, Matrix3x3
from .constants import wgs84, D2R, R2D


def skew_symmetric(v: Vector3) -> Matrix3x3:
    """
    构造反对称矩阵
    
    Args:
        v: 3维向量
    
    Returns:
        3x3反对称矩阵
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def quat2dcm(q: Quaternion) -> Matrix3x3:
    """
    四元数转方向余弦矩阵
    
    Args:
        q: 四元数 (b系到n系)
    
    Returns:
        3x3旋转矩阵 C_b^n
    """
    w, x, y, z = q.w, q.x, q.y, q.z
    
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ])


def dcm2quat(C: Matrix3x3) -> Quaternion:
    """
    方向余弦矩阵转四元数
    
    Args:
        C: 3x3旋转矩阵
    
    Returns:
        四元数
    """
    trace = np.trace(C)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (C[2, 1] - C[1, 2]) * s
        y = (C[0, 2] - C[2, 0]) * s
        z = (C[1, 0] - C[0, 1]) * s
    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2])
        w = (C[2, 1] - C[1, 2]) / s
        x = 0.25 * s
        y = (C[0, 1] + C[1, 0]) / s
        z = (C[0, 2] + C[2, 0]) / s
    elif C[1, 1] > C[2, 2]:
        s = 2.0 * np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2])
        w = (C[0, 2] - C[2, 0]) / s
        x = (C[0, 1] + C[1, 0]) / s
        y = 0.25 * s
        z = (C[1, 2] + C[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1])
        w = (C[1, 0] - C[0, 1]) / s
        x = (C[0, 2] + C[2, 0]) / s
        y = (C[1, 2] + C[2, 1]) / s
        z = 0.25 * s
    
    return Quaternion(w=w, x=x, y=y, z=z).normalize()


def quat2euler(q: Quaternion) -> Vector3:
    """
    四元数转欧拉角 (NED坐标系)
    
    Args:
        q: 四元数
    
    Returns:
        欧拉角 [roll, pitch, yaw] (rad)
    """
    C = quat2dcm(q)
    return dcm2euler(C)


def dcm2euler(C: Matrix3x3) -> Vector3:
    """
    方向余弦矩阵转欧拉角 (NED坐标系)
    
    Args:
        C: 旋转矩阵
    
    Returns:
        欧拉角 [roll, pitch, yaw] (rad)
    """
    # Roll (绕x轴)
    roll = np.arctan2(C[2, 1], C[2, 2])
    
    # Pitch (绕y轴)
    pitch = np.arcsin(-C[2, 0])
    
    # Yaw (绕z轴)
    yaw = np.arctan2(C[1, 0], C[0, 0])
    
    return np.array([roll, pitch, yaw])


def euler2quat(euler: Vector3) -> Quaternion:
    """
    欧拉角转四元数
    
    Args:
        euler: 欧拉角 [roll, pitch, yaw] (rad)
    
    Returns:
        四元数
    """
    roll, pitch, yaw = euler
    
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return Quaternion(w=w, x=x, y=y, z=z).normalize()


def euler2dcm(euler: Vector3) -> Matrix3x3:
    """
    欧拉角转方向余弦矩阵
    
    Args:
        euler: 欧拉角 [roll, pitch, yaw] (rad)
    
    Returns:
        旋转矩阵
    """
    roll, pitch, yaw = euler
    
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    return np.array([
        [cp*cy, -cr*sy + sr*sp*cy, sr*sy + cr*sp*cy],
        [cp*sy, cr*cy + sr*sp*sy, -sr*cy + cr*sp*sy],
        [-sp, sr*cp, cr*cp]
    ])


def rv2quat(rv: Vector3) -> Quaternion:
    """
    旋转向量转四元数
    
    Args:
        rv: 旋转向量 [rad]
    
    Returns:
        四元数
    """
    angle = np.linalg.norm(rv)
    
    if angle < 1e-8:
        # 小角度近似
        return Quaternion(w=1.0, x=rv[0]/2, y=rv[1]/2, z=rv[2]/2)
    
    axis = rv / angle
    half_angle = angle / 2.0
    s = np.sin(half_angle)
    
    return Quaternion(
        w=np.cos(half_angle),
        x=axis[0] * s,
        y=axis[1] * s,
        z=axis[2] * s
    )


def quat_rotate_vector(q: Quaternion, v: Vector3) -> Vector3:
    """
    使用四元数旋转向量
    
    Args:
        q: 旋转四元数
        v: 待旋转向量
    
    Returns:
        旋转后的向量
    """
    # 将向量转换为四元数 (w=0)
    v_quat = Quaternion(w=0, x=v[0], y=v[1], z=v[2])
    
    # q * v * q^(-1)
    result = q.multiply(v_quat).multiply(q.conjugate())
    
    return np.array([result.x, result.y, result.z])


def ecef2llh(xyz: Vector3) -> Vector3:
    """
    ECEF坐标转大地坐标
    
    Args:
        xyz: ECEF坐标 [x, y, z] (m)
    
    Returns:
        大地坐标 [lat(rad), lon(rad), height(m)]
    """
    x, y, z = xyz
    
    # 经度
    lon = np.arctan2(y, x)
    
    # 迭代计算纬度和高度
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - wgs84.e2))
    
    for _ in range(5):  # 迭代5次
        N = wgs84.a / np.sqrt(1 - wgs84.e2 * np.sin(lat)**2)
        height = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - wgs84.e2 * N / (N + height)))
    
    N = wgs84.a / np.sqrt(1 - wgs84.e2 * np.sin(lat)**2)
    height = p / np.cos(lat) - N
    
    return np.array([lat, lon, height])


def llh2ecef(llh: Vector3) -> Vector3:
    """
    大地坐标转ECEF坐标
    
    Args:
        llh: 大地坐标 [lat(rad), lon(rad), height(m)]
    
    Returns:
        ECEF坐标 [x, y, z] (m)
    """
    lat, lon, h = llh
    
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    N = wgs84.a / np.sqrt(1 - wgs84.e2 * sin_lat**2)
    
    x = (N + h) * cos_lat * cos_lon
    y = (N + h) * cos_lat * sin_lon
    z = (N * (1 - wgs84.e2) + h) * sin_lat
    
    return np.array([x, y, z])


def get_Cen(lat: float, lon: float) -> Matrix3x3:
    """
    计算ECEF到NED的旋转矩阵
    
    Args:
        lat: 纬度 [rad]
        lon: 经度 [rad]
    
    Returns:
        C_e^n 旋转矩阵
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    return np.array([
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [-sin_lon, cos_lon, 0],
        [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat]
    ])


def ecef2enu(xyz: Vector3, ref_llh: Vector3) -> Vector3:
    """
    ECEF坐标转ENU坐标
    
    Args:
        xyz: ECEF坐标 [m]
        ref_llh: 参考点大地坐标 [lat(rad), lon(rad), height(m)]
    
    Returns:
        ENU坐标 [east, north, up] (m)
    """
    ref_xyz = llh2ecef(ref_llh)
    delta_xyz = xyz - ref_xyz
    
    lat, lon, _ = ref_llh
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    # ECEF到ENU的旋转矩阵
    C_e2enu = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    
    return C_e2enu @ delta_xyz


def enu2ecef(enu: Vector3, ref_llh: Vector3) -> Vector3:
    """
    ENU坐标转ECEF坐标
    
    Args:
        enu: ENU坐标 [east, north, up] (m)
        ref_llh: 参考点大地坐标 [lat(rad), lon(rad), height(m)]
    
    Returns:
        ECEF坐标 [x, y, z] (m)
    """
    lat, lon, _ = ref_llh
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    
    # ENU到ECEF的旋转矩阵
    C_enu2e = np.array([
        [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
        [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
        [0, cos_lat, sin_lat]
    ])
    
    delta_xyz = C_enu2e @ enu
    ref_xyz = llh2ecef(ref_llh)
    
    return ref_xyz + delta_xyz


def ned2enu(ned: Vector3) -> Vector3:
    """NED转ENU"""
    return np.array([ned[1], ned[0], -ned[2]])


def enu2ned(enu: Vector3) -> Vector3:
    """ENU转NED"""
    return np.array([enu[1], enu[0], -enu[2]])


def geodist(sat_pos: np.ndarray, rcv_pos: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算几何距离和视线向量
    
    Args:
        sat_pos: 卫星位置 [x, y, z] (m)
        rcv_pos: 接收机位置 [x, y, z] (m)
    
    Returns:
        r: 几何距离 (m)
        e: 单位视线向量 [ex, ey, ez]
    """
    # 计算相对位置向量
    dx = sat_pos - rcv_pos
    
    # 计算距离
    r = np.linalg.norm(dx)
    
    # 计算单位视线向量
    if r > 0.0:
        e = dx / r
    else:
        e = np.zeros(3)
    
    return r, e

