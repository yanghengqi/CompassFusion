"""
GNSS卫星位置计算和观测模型
"""

import numpy as np
from typing import Tuple, Optional, Union
from ..core.gnss_types import GlonassEphemeris, NavigationData, SatelliteObservation
from ..core.constants import wgs84, CLIGHT, D2R
from ..core.transforms import ecef2llh, llh2ecef


class SatellitePositionCalculator:
    """卫星位置计算器（基于广播星历）"""
    
    # 系统常数定义（参考COMPASS ephemeris.c）
    MU_GPS = 3.9860050E14      # GPS引力常数
    MU_GAL = 3.986004418E14    # Galileo引力常数
    MU_CMP = 3.986004418E14    # BDS引力常数
    MU_GLO = 3.9860044E14      # GLONASS引力常数
    
    OMGE_GPS = 7.2921151467E-5  # GPS地球自转角速度
    OMGE_GAL = 7.2921151467E-5  # Galileo地球自转角速度
    OMGE_CMP = 7.292115E-5      # BDS地球自转角速度
    OMGE_GLO = 7.292115E-5      # GLONASS地球自转角速度

    # BDS GEO 轨道倾角常数（对齐 RTKLIB/COMPASS ephemeris.c）
    COS_5 = 0.9961946980917455  # cos(5°)
    SIN_5 = 0.08715574274765817 # sin(5°)

    # GLONASS 数值积分（对齐 RTKLIB ephemeris.c deq/glorbit/geph2pos）
    RE_GLO = 6378136.0
    J2_GLO = 1.0826257E-3
    TSTEP_GLO = 60.0

    @staticmethod
    def _pz90_to_itrf2008_great(xyz_pz: np.ndarray) -> np.ndarray:
        """
        PZ-90.11 → ITRF2008（与 GREAT-PVT LibGnut gnavglo.cpp::_pos 226–241 行一致）。
        广播积分得到的位置在 PZ-90，与 GPS/BDS/Galileo 的 WGS84/ITRF 框架混算前需变换。
        """
        r = np.asarray(xyz_pz, dtype=float).reshape(3)
        t_vec = np.array([-0.003, -0.001, 0.0], dtype=float)
        mas = (1.0 / 36.0e5) * D2R
        rot = np.array(
            [
                [1.0, 0.002 * mas, 0.042 * mas],
                [-0.002 * mas, 1.0, 0.019 * mas],
                [-0.042 * mas, -0.019 * mas, 1.0],
            ],
            dtype=float,
        )
        return (t_vec + rot @ r).reshape(3)

    @staticmethod
    def _deq_glo(x: np.ndarray, acc: np.ndarray) -> np.ndarray:
        r2 = float(np.dot(x[0:3], x[0:3]))
        xdot = np.zeros(6)
        if r2 <= 0.0:
            return xdot
        r3 = r2 * np.sqrt(r2)
        mu = SatellitePositionCalculator.MU_GLO
        omge = SatellitePositionCalculator.OMGE_GLO
        re = SatellitePositionCalculator.RE_GLO
        j2 = SatellitePositionCalculator.J2_GLO
        omg2 = omge * omge
        a = 1.5 * j2 * mu * (re * re) / r2 / r3
        b = 5.0 * x[2] * x[2] / r2
        c = -mu / r3 - a * (1.0 - b)
        xdot[0] = x[3]
        xdot[1] = x[4]
        xdot[2] = x[5]
        xdot[3] = (c + omg2) * x[0] + 2.0 * omge * x[4] + acc[0]
        xdot[4] = (c + omg2) * x[1] - 2.0 * omge * x[3] + acc[1]
        xdot[5] = (c - 2.0 * a) * x[2] + acc[2]
        return xdot

    @staticmethod
    def _glorbit_glo(t: float, x: np.ndarray, acc: np.ndarray) -> None:
        k1 = SatellitePositionCalculator._deq_glo(x, acc)
        w = x + k1 * (t / 2.0)
        k2 = SatellitePositionCalculator._deq_glo(w, acc)
        w = x + k2 * (t / 2.0)
        k3 = SatellitePositionCalculator._deq_glo(w, acc)
        w = x + k3 * t
        k4 = SatellitePositionCalculator._deq_glo(w, acc)
        x[:] += (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (t / 6.0)

    @staticmethod
    def _geph2pos(
        geph: GlonassEphemeris, time: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        t = float(time - geph.toe)
        dts = float(-geph.taun + geph.gamn * t)
        x = np.zeros(6, dtype=float)
        x[0:3] = np.asarray(geph.pos, dtype=float)
        x[3:6] = np.asarray(geph.vel, dtype=float)
        acc = np.asarray(geph.acc, dtype=float)
        ts = SatellitePositionCalculator.TSTEP_GLO
        tt = -ts if t < 0.0 else ts
        t_rem = t
        while abs(t_rem) > 1e-9:
            if abs(t_rem) < ts:
                tt = t_rem
            SatellitePositionCalculator._glorbit_glo(tt, x, acc)
            t_rem -= tt
        pos_wgs = SatellitePositionCalculator._pz90_to_itrf2008_great(x[0:3])
        return pos_wgs, x[3:6].copy(), dts
    
    @staticmethod
    def compute_satellite_position(
        nav: Union[NavigationData, GlonassEphemeris],
        time: float,
        system: str = 'GPS'
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算卫星位置和钟差（支持 GPS/Galileo/BDS/QZSS 及 GLONASS 广播星历）
        
        Args:
            nav: 导航星历（Kepler 或 GlonassEphemeris）
            time: 系统时间 [s]（GPST 连续秒，与其它系统一致）
            system: 卫星系统（Kepler 时用；GLO 时以 nav 类型为准）
        
        Returns:
            (位置ECEF [m], 速度ECEF [m/s], 钟差 [s])
        
        参考: COMPASS/RTKLIB ephemeris.c eph2pos() / geph2pos()
        """
        if isinstance(nav, GlonassEphemeris):
            return SatellitePositionCalculator._geph2pos(nav, time)
        calc = SatellitePositionCalculator()
        
        # 根据系统选择常数
        if system == 'Galileo':
            mu = calc.MU_GAL
            omge = calc.OMGE_GAL
        elif system == 'BDS':
            mu = calc.MU_CMP
            omge = calc.OMGE_CMP
        else:  # GPS, QZSS
            mu = calc.MU_GPS
            omge = calc.OMGE_GPS
        
        return calc._compute_kepler_orbit(nav, time, mu, omge, system)
    
    @staticmethod
    def compute_satellite_position_gps(
        nav: NavigationData,
        time: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        计算GPS卫星位置和钟差（向后兼容）
        
        Args:
            nav: 导航星历
            time: GPS时间 [s]
        
        Returns:
            (位置ECEF [m], 速度ECEF [m/s], 钟差 [s])
        """
        calc = SatellitePositionCalculator()
        return calc.compute_satellite_position(nav, time, 'GPS')
    
    def _compute_kepler_orbit(
        self,
        nav: NavigationData,
        time: float,
        mu: float,
        omge: float,
        system: str
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Kepler轨道计算（GPS/Galileo/BDS/QZSS通用）
        
        参考: COMPASS ephemeris.c eph2pos()
        """
        # 时间校正
        dt = time - nav.toe
        
        # 卫星钟差（含相对论改正）
        dt_clk = time - nav.toc
        sat_clock = nav.af0 + nav.af1 * dt_clk + nav.af2 * dt_clk**2
        
        # 轨道参数
        a = nav.sqrt_a ** 2  # 长半轴
        n0 = np.sqrt(mu / a**3)  # 平均角速度
        n = n0 + nav.delta_n  # 改正后的平均角速度
        
        # 平近点角
        M = nav.M0 + n * dt
        
        # 迭代求解偏近点角E
        E = M
        for _ in range(10):
            E_old = E
            E = M + nav.e * np.sin(E)
            if abs(E - E_old) < 1e-12:
                break
        
        # 真近点角
        sin_E = np.sin(E)
        cos_E = np.cos(E)
        sqrt_1_e2 = np.sqrt(1 - nav.e**2)
        
        sin_v = sqrt_1_e2 * sin_E / (1 - nav.e * cos_E)
        cos_v = (cos_E - nav.e) / (1 - nav.e * cos_E)
        v = np.arctan2(sin_v, cos_v)
        
        # 纬度幅角
        phi = v + nav.omega
        sin_2phi = np.sin(2 * phi)
        cos_2phi = np.cos(2 * phi)
        
        # 摄动改正
        du = nav.cuc * cos_2phi + nav.cus * sin_2phi  # 纬度幅角改正
        dr = nav.crc * cos_2phi + nav.crs * sin_2phi  # 轨道半径改正
        di = nav.cic * cos_2phi + nav.cis * sin_2phi  # 轨道倾角改正
        
        # 改正后的参数
        u = phi + du  # 纬度幅角
        r = a * (1 - nav.e * cos_E) + dr  # 轨道半径
        i = nav.i0 + nav.i_dot * dt + di  # 轨道倾角
        
        # 升交点赤经
        # toe_sow 须为系统自身时间的周内秒（Ω₀ 的参考基准）：
        #   GPS/GAL/QZSS: GPST 周内秒
        #   BDS:          BDT 周内秒 = (nav.toe - 14) % 604800
        #                 因为 nav.toe 以 GPST 存储，GPST = BDT + 14 s
        if system == 'BDS':
            toe_sow = (nav.toe - 14.0) % 604800.0
        else:
            toe_sow = nav.toe % 604800.0
        # BDS GEO（C01-05, C59+）：采用特殊坐标变换（对齐 ephemeris.c eph2pos）
        is_bds_geo = (system == 'BDS') and (nav.sat_id <= 5 or nav.sat_id >= 59)
        if is_bds_geo:
            omega = nav.omega0 + nav.omega_dot * dt - omge * toe_sow
        else:
            omega = nav.omega0 + (nav.omega_dot - omge) * dt - omge * toe_sow
        
        # 轨道平面坐标
        x_orb = r * np.cos(u)
        y_orb = r * np.sin(u)
        
        # ECEF坐标
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        cos_i = np.cos(i)
        sin_i = np.sin(i)

        if is_bds_geo:
            # GEO: 先按轨道平面→地固的中间量 xg/yg/zg，再做地球自转与5°倾角变换
            xg = x_orb * cos_omega - y_orb * cos_i * sin_omega
            yg = x_orb * sin_omega + y_orb * cos_i * cos_omega
            zg = y_orb * sin_i

            sino = np.sin(omge * dt)
            coso = np.cos(omge * dt)

            x = xg * coso + yg * sino * self.COS_5 + zg * sino * self.SIN_5
            y = -xg * sino + yg * coso * self.COS_5 + zg * coso * self.SIN_5
            z = -yg * self.SIN_5 + zg * self.COS_5
        else:
            x = x_orb * cos_omega - y_orb * cos_i * sin_omega
            y = x_orb * sin_omega + y_orb * cos_i * cos_omega
            z = y_orb * sin_i
        
        position = np.array([x, y, z])
        
        # 相对论改正（影响钟差）
        # 参考: ICD-GPS-200, 20.3.3.3.3.1
        relativistic_correction = -2.0 * np.sqrt(mu * a) * nav.e * sin_E / (CLIGHT ** 2)
        sat_clock += relativistic_correction
        
        # 计算速度（简化）
        E_dot = n / (1 - nav.e * cos_E)
        v_dot = sqrt_1_e2 * E_dot / (1 - nav.e * cos_E)
        
        u_dot = v_dot + 2 * (nav.cuc * cos_2phi - nav.cus * sin_2phi) * v_dot
        r_dot = a * nav.e * sin_E * E_dot + 2 * (nav.crc * cos_2phi - nav.crs * sin_2phi) * v_dot
        i_dot_corrected = nav.i_dot + 2 * (nav.cic * cos_2phi - nav.cis * sin_2phi) * v_dot
        omega_dot = nav.omega_dot - omge
        
        vx_orb = r_dot * np.cos(u) - r * u_dot * np.sin(u)
        vy_orb = r_dot * np.sin(u) + r * u_dot * np.cos(u)
        
        vx = vx_orb * cos_omega - vy_orb * cos_i * sin_omega + y_orb * sin_i * sin_omega * i_dot_corrected - (x_orb * sin_omega + y_orb * cos_i * cos_omega) * omega_dot
        vy = vx_orb * sin_omega + vy_orb * cos_i * cos_omega - y_orb * sin_i * cos_omega * i_dot_corrected + (x_orb * cos_omega - y_orb * cos_i * sin_omega) * omega_dot
        vz = vy_orb * sin_i + y_orb * cos_i * i_dot_corrected
        
        velocity = np.array([vx, vy, vz])
        
        return position, velocity, sat_clock


class GNSSObservationModel:
    """GNSS观测模型"""
    
    @staticmethod
    def compute_elevation_azimuth(
        sat_pos: np.ndarray,
        user_pos: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算卫星高度角和方位角
        
        Args:
            sat_pos: 卫星ECEF位置 [m]
            user_pos: 用户LLH位置 [lat(rad), lon(rad), h(m)]
        
        Returns:
            (高度角 [rad], 方位角 [rad])
        """
        # 用户ECEF位置
        user_xyz = llh2ecef(user_pos)
        
        # 卫星到用户的向量（ECEF）
        los_ecef = sat_pos - user_xyz
        
        # 转换到ENU坐标系
        lat, lon = user_pos[0], user_pos[1]
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
        
        los_enu = C_e2enu @ los_ecef
        
        # 高度角和方位角
        E, N, U = los_enu
        elevation = np.arctan2(U, np.sqrt(E**2 + N**2))
        azimuth = np.arctan2(E, N)
        
        if azimuth < 0:
            azimuth += 2 * np.pi
        
        return elevation, azimuth
    
    @staticmethod
    def compute_geometric_range(
        sat_pos: np.ndarray,
        user_pos: np.ndarray,
        earth_rotation_correction: bool = True
    ) -> float:
        """
        计算几何距离（考虑地球自转）
        
        Args:
            sat_pos: 卫星ECEF位置 [m]
            user_pos: 用户ECEF位置 [m]
            earth_rotation_correction: 是否进行地球自转改正
        
        Returns:
            几何距离 [m]
        """
        # 信号传播时间
        dx = sat_pos - user_pos
        tau = np.linalg.norm(dx) / CLIGHT
        
        if earth_rotation_correction:
            # Sagnac效应改正
            omega_e = wgs84.omega_ie
            sat_pos_corrected = np.array([
                sat_pos[0] * np.cos(omega_e * tau) + sat_pos[1] * np.sin(omega_e * tau),
                -sat_pos[0] * np.sin(omega_e * tau) + sat_pos[1] * np.cos(omega_e * tau),
                sat_pos[2]
            ])
            dx = sat_pos_corrected - user_pos
        
        return np.linalg.norm(dx)
    
    @staticmethod
    def compute_tropospheric_delay(elevation: float, height: float = 0.0) -> float:
        """
        对流层延迟（Saastamoinen模型）
        
        Args:
            elevation: 高度角 [rad]
            height: 用户高度 [m]
        
        Returns:
            对流层延迟 [m]
        """
        if elevation <= 0:
            return 0.0
        
        # 简化的Saastamoinen模型
        P = 1013.25 * (1 - 2.2557e-5 * height) ** 5.2568  # 大气压 [mbar]
        T = 15.0 - 6.5e-3 * height + 273.15  # 温度 [K]
        e = 6.108 * np.exp((17.15 * (T - 273.15)) / (234.9 + (T - 273.15))) * 0.5  # 水汽压 [mbar]
        
        sin_elev = np.sin(elevation)
        
        # 天顶延迟
        trop_dry = 0.002277 * P / sin_elev
        trop_wet = 0.002277 * (1255.0 / T + 0.05) * e / sin_elev
        
        return trop_dry + trop_wet
    
    @staticmethod
    def compute_ionospheric_delay_klobuchar(
        elevation: float,
        azimuth: float,
        user_lat: float,
        user_lon: float,
        time: float,
        iono_params: Optional[np.ndarray] = None
    ) -> float:
        """
        电离层延迟（Klobuchar模型）
        
        Args:
            elevation: 高度角 [rad]
            azimuth: 方位角 [rad]
            user_lat: 用户纬度 [rad]
            user_lon: 用户经度 [rad]
            time: GPS时间 [s]
            iono_params: 电离层参数 [α0, α1, α2, α3, β0, β1, β2, β3]
        
        Returns:
            电离层延迟 [m] (L1频点)
        """
        if elevation <= 0:
            return 0.0
        
        # 默认参数
        if iono_params is None:
            iono_params = np.zeros(8)
        
        # 简化：返回简单估计值
        # 完整实现需要Klobuchar公式
        F = 1.0 + 16.0 * (0.53 - elevation / np.pi) ** 3
        delay = 5.0 / np.sin(elevation) * F  # 约5-50m
        
        return delay
