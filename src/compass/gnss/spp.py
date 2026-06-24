"""
SPP (Standard Point Positioning) - Single Point Positioning
基于COMPASS pntpos.c的完整实现
"""

import os
from bisect import bisect_left
from dataclasses import replace
import numpy as np
from typing import List, Tuple, Optional
from ..core.gnss_types import (
    GlonassEphemeris,
    GNSSRawObservation,
    NavigationData,
    NavRecord,
    SatelliteObservation,
)
from ..core.constants import CLIGHT, wgs84
from ..gnss.satellite import SatellitePositionCalculator
from ..core.transforms import ecef2llh, llh2ecef
from .ionosphere import ionocorr
from .bias_sinex import BiasSinexDCB


def _lorentz_inner_4(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - a[3] * b[3])


def bancroft_solve_lorentz(BBpass: np.ndarray) -> Optional[np.ndarray]:
    """
    G-Nut gbancroft() 的 Lorentz 内积闭式解（对齐 LibGnut/gmodels/gbancroft.cpp）。
    BBpass: (n, 4)，每行 [xs, ys, zs, P4]，P4 与 GREAT 中 P3+sat_clk 同量级（米）。
    返回长度 4 的向量 [x,y,z,b]，失败返回 None。
    """
    BBpass = np.asarray(BBpass, dtype=float)
    if BBpass.ndim != 2 or BBpass.shape[1] != 4 or BBpass.shape[0] < 4:
        return None
    pos = np.zeros(4, dtype=float)
    OMEGA = 7.2921151467e-5
    for it in (1, 2):
        BB = BBpass.copy()
        mm = BB.shape[0]
        for ii in range(mm):
            xx, yy = BB[ii, 0], BB[ii, 1]
            traveltime = 0.072
            if it > 1:
                zz = BB[ii, 2]
                rho2 = (xx - pos[0]) ** 2 + (yy - pos[1]) ** 2 + (zz - pos[2]) ** 2
                if rho2 < 0:
                    return None
                rho = float(np.sqrt(rho2))
                traveltime = rho / CLIGHT
            angle = traveltime * OMEGA
            cosa, sina = np.cos(angle), np.sin(angle)
            BB[ii, 0] = cosa * xx + sina * yy
            BB[ii, 1] = -sina * xx + cosa * yy
        if mm > 4:
            btb = BB.T @ BB
            try:
                BBB = np.linalg.solve(btb, BB.T)
            except np.linalg.LinAlgError:
                return None
        else:
            try:
                BBB = np.linalg.inv(BB)
            except np.linalg.LinAlgError:
                return None
        alpha = np.zeros(mm, dtype=float)
        for ii in range(mm):
            bi = BB[ii, :]
            alpha[ii] = _lorentz_inner_4(bi, bi) * 0.5
        ee = np.ones(mm, dtype=float)
        BBBe = BBB @ ee
        BBBalpha = BBB @ alpha
        aa = _lorentz_inner_4(BBBe, BBBe)
        bb = _lorentz_inner_4(BBBe, BBBalpha) - 1.0
        cc = _lorentz_inner_4(BBBalpha, BBBalpha)
        root2 = bb * bb - aa * cc
        if root2 < 0.0 or abs(aa) < 1e-18:
            return None
        root = float(np.sqrt(root2))
        sol1 = (-bb - root) / aa * BBBe + BBBalpha
        sol2 = (-bb + root) / aa * BBBe + BBBalpha
        omc = np.zeros(2, dtype=float)
        for pp in range(2):
            hlp = (sol1 if pp == 0 else sol2).copy()
            hlp[3] = -hlp[3]
            tm = (
                (BB[0, 0] - hlp[0]) ** 2
                + (BB[0, 1] - hlp[1]) ** 2
                + (BB[0, 2] - hlp[2]) ** 2
            )
            if tm < 0:
                return None
            omc[pp] = BB[0, 3] - float(np.sqrt(tm)) - hlp[3]
        if abs(omc[0]) > abs(omc[1]):
            pos = sol2.copy()
        else:
            pos = sol1.copy()
    return pos


class SPPSolver:
    """
    SPP 单点定位解算器。

    参考 COMPASS pntpos；多系统码/钟建模默认对齐 GREAT-PVT/LibGnut：
    各系统 ISB、BDS 频间 IFB、GLONASS 为 GLO_ISB + 每星 GLO_IFB（FDMA 码残差按星吸收）；
    GLO 广播位置在 satellite._geph2pos 中经 PZ-90.11→ITRF2008。
    设 SPP_RTKLIB_PNTPOS_ISB=1 或 SPP_GLO_ISB_MODE=rtklib 时退化为 RTKLIB 式「仅 GLO–GPS 一项 x[4]」；
    SPP_GLO_ISB_MODE=auto 时按星 IFB 失败后再试该单参数（部分场景可能收敛但偏差大，慎用）。
    """
    
    # 常数定义（参考pntpos.c）
    MAXITR = 10  # 最大迭代次数
    ERR_ION = 5.0  # 电离层延迟标准差 (m)
    ERR_TROP = 3.0  # 对流层延迟标准差 (m)
    ERR_SAAS = 0.3  # Saastamoinen模型误差标准差 (m)
    ERR_CBIAS = 0.3  # 码偏差误差标准差 (m)
    OMGE = 7.2921151467e-5  # 地球自转角速度 (rad/s) - WGS84
    # 收敛后残差剔除阈值（m）
    # 之前用 5m 会在多系统码观测下过于激进，容易导致反复重试并把位置“锁死”在初值。
    RES_EXC_TH = 30.0
    # 迭代更新步长门限：防止多系统未建模偏差导致坐标“飞走”
    MAX_DX_POS = 100.0  # m per iteration
    MAX_DX_CLK = 1e5    # m per iteration

    # KF 时间更新过程噪声（随机游走，单位 m^2/s）
    Q_POS = 1e-2      # 位置
    Q_CLK = 1e4       # 钟差（m）
    Q_ISB = 1e2       # 系统间偏差（m）
    Q_IFB = 1e1       # 频间码偏差（m）
    # 与 compass-master/src/LibGnss/pntpos.c 对齐的“系统钟差/ISB”参数个数：
    # x,y,z + (GPS钟差 + GLO/GAL/BDS/QZS/BD3 相对GPS的ISB)
    NUM_SYS = 6
    # 借鉴 GREAT-PVT：BDS 频间 IFB（固定维）；GLONASS 另按历元追加「每星 IFB」（对齐 gsppmodel isbCorrection: GLO_ISB + GLO_IFB_sat）
    N_IFB = 2
    NX_BASE = 3 + NUM_SYS + N_IFB
    NX = NX_BASE  # 兼容旧代码：无 GLO 时状态维

    IDX_IFB_C7 = 3 + NUM_SYS + 0  # BDS 使用 C7* 作为第二频点
    IDX_IFB_C6 = 3 + NUM_SYS + 1  # BDS 使用 C6* 作为第二频点

    @staticmethod
    def _sys_index(system: str, sat_id: int) -> int:
        """与 pntpos.c 的 satsysidx() 对齐的系统索引。

        0: GPS
        1: GLO
        2: GAL
        3: BDS(BD2)
        4: QZS
        5: BDS(BD3)  (这里用 sat_id>18 近似区分)
        """
        if system == 'G':
            return 0
        if system == 'R':
            return 1
        if system == 'E':
            return 2
        if system == 'J':
            return 4
        if system == 'C':
            # GREAT-PVT 在 SPP/PPP 中通常把 BDS 作为同一系统处理；
            # 这里先不拆 BD2/BD3，避免单历元多 ISB 参数导致病态与跑飞。
            return 3
        return -1

    @staticmethod
    def _freq_pair_hz(system: str) -> Tuple[float, float]:
        """返回用于电离层无关组合(IFLC)的 (f1,f2) 频率对（Hz）。

        近似对齐 RTKLIB 默认：
        - GPS/QZSS: L1/L2
        - Galileo : E1/E5b
        - BDS     : B1I/B2I
        """
        if system in ('G', 'J'):  # GPS/QZSS
            return 1575.42e6, 1227.60e6
        if system == 'E':  # Galileo
            return 1575.42e6, 1207.14e6
        if system == 'C':  # BDS
            return 1561.098e6, 1207.14e6
        if system == 'R':  # GLO: 频点依赖通道号，这里不做 IFLC
            return 0.0, 0.0
        return 0.0, 0.0

    @staticmethod
    def _freq_L1_hz(system: str) -> float:
        """用于单频电离层缩放的近似频点（Hz）。

        对齐 pntpos.c 的做法：在 ionocorr() 得到的延迟基础上，按 (f_ref/f)^2 缩放到实际观测频点。
        这里用常用信号近似，不处理 GLONASS 频点偏移。
        """
        if system in ('G', 'J'):  # GPS/QZSS L1
            return 1575.42e6
        if system == 'E':         # Galileo E1
            return 1575.42e6
        if system == 'C':         # BDS B1I/B1C 近似按 B1I
            return 1561.098e6
        if system == 'R':         # GLO: 不在本SPP混合里重点支持
            return 1602.0e6
        return 0.0

    @staticmethod
    def _bds_rinex_code_to_freq_hz(code: Optional[str]) -> Optional[float]:
        """RINEX 伪距码 → 载波频率 (Hz)，用于 IF 的 gamma 与电离层缩放。"""
        if not code or len(code) < 2:
            return None
        c = code.upper()
        if c in ("C2I", "C2X", "C2Q", "C2P"):
            return 1561.098e6
        if c.startswith("C1"):
            return 1575.42e6
        if c in ("C7I", "C7X", "C7Q", "C7D", "C7P", "C7Z"):
            return 1207.14e6
        if c.startswith("C5"):
            return 1176.45e6
        if c.startswith("C6"):
            return 1268.52e6
        return None

    @staticmethod
    def _bds_iflc_prange_nav(
        P1: float, P2: float, code1: Optional[str], code2: Optional[str], nav: NavigationData
    ) -> Optional[float]:
        """BDS 双频 IF + 广播群延迟（借鉴 GREAT-PVT：BDS TGD 以 B3 为基准）。

        - nav.tgd  ≈ TGD(B1/B3) [s]
        - nav.tgd2 ≈ TGD(B2/B3) [s]  (B2=B2I/B2b)
        """
        f1 = SPPSolver._bds_rinex_code_to_freq_hz(code1)
        f2 = SPPSolver._bds_rinex_code_to_freq_hz(code2)
        if f1 is None or f2 is None or f1 <= 0.0 or f2 <= 0.0:
            return None

        # GREAT-PVT: Pc = alpha*(P1-b1) + beta*(P2-b2)
        f1_2 = f1 * f1
        f2_2 = f2 * f2
        den = f1_2 - f2_2
        if den == 0.0:
            return None
        alpha = f1_2 / den
        beta = -f2_2 / den

        def _bias_m(code: Optional[str]) -> Optional[float]:
            c = (code or "").upper()
            if c.startswith("C6"):  # B3
                return 0.0
            if c.startswith("C7"):  # B2I/B2b
                return nav.tgd2 * CLIGHT
            if c.startswith("C1") or c.startswith("C2"):  # B1C/B1I
                return nav.tgd * CLIGHT
            if c.startswith("C5"):  # B2a: no broadcast TGD w.r.t B3 here
                return None
            return None

        b1 = _bias_m(code1)
        b2 = _bias_m(code2)
        if b1 is None or b2 is None:
            return None

        return alpha * (P1 - b1) + beta * (P2 - b2)

    @staticmethod
    def _prange_iflc(
        system: str,
        P1: float,
        P2: float,
        nav: NavRecord,
        sat_obs: Optional[SatelliteObservation] = None,
    ) -> Optional[float]:
        """对齐 pntpos.c 的 iono-free 伪距组合（含广播TGD/BGD修正的关键项）。

        返回 IFLC 伪距（m），失败返回 None。
        """
        # GLONASS FDMA：按频率号 k 计算 G1/G2（Hz），无广播 TGD 进入 IF 组合
        if system == 'R':
            if not isinstance(nav, GlonassEphemeris):
                return None
            f1 = 1602.0e6 + nav.frq * 562500.0
            f2 = 1246.0e6 + nav.frq * 437500.0
            if f1 <= 0.0 or f2 <= 0.0:
                return None
            gamma = (f1 / f2) ** 2
            return (P2 - gamma * P1) / (1.0 - gamma)

        f1, f2 = SPPSolver._freq_pair_hz(system)
        if f1 <= 0.0 or f2 <= 0.0:
            return None
        gamma = (f1 / f2) ** 2

        # GPS/QZSS: 直接 IF 组合（广播TGD通常在其他地方一致处理，这里保持与 pntpos.c 类似的轻量方式）
        if system in ('G', 'J'):
            return (P2 - gamma * P1) / (1.0 - gamma)

        # Galileo: E1/E5b。pntpos.c 对 P2 做 BGD_E5aE1 - BGD_E5bE1 修正
        if system == 'E':
            P2_corr = P2 - (nav.tgd - nav.tgd2) * CLIGHT
            return (P2_corr - gamma * P1) / (1.0 - gamma)

        # BDS: 按实际选用的 RINEX 码计算 f1/f2 与 TGD 组合；失败则退回 B1I/B2I 常数频比
        if system == 'C':
            if sat_obs is not None and sat_obs.code_pr1 and sat_obs.code_pr2:
                pr = SPPSolver._bds_iflc_prange_nav(
                    P1, P2, sat_obs.code_pr1, sat_obs.code_pr2, nav
                )
                if pr is not None and np.isfinite(pr):
                    return pr
            b1 = nav.tgd
            b2 = nav.tgd2
            return ((P2 - gamma * P1) - (b2 - gamma * b1) * CLIGHT) / (1.0 - gamma)

        return None
    
    def __init__(
        self,
        elev_mask: float = 15.0,
        ion_gps: Optional[np.ndarray] = None,
        ion_bds: Optional[np.ndarray] = None,
        glo_ifb_ema_beta: Optional[float] = None,
    ):
        """
        初始化SPP解算器

        Args:
            elev_mask: 高度角截止角 (度)
            ion_gps: 广播 GPS/Galileo/GLONASS Klobuchar 系数，长度 8（α0..α3,β0..β3），
                通常由 RINEX 导航头 GPSA+GPSB 填入；None 时用内置默认参数。
            ion_bds: 广播 BDS 电离层系数 8 元组；None 时 BDS 退回 GPS 模型或默认。
            glo_ifb_ema_beta: GLONASS 偏差跨历元平滑系数 β∈[0,1]；None 时读环境变量
                SPP_GLO_EMA（默认 0.12）。β=0 关闭。仅用 obs+nav，改善 FDMA 单历元可估性。
        """
        self.elev_mask = np.deg2rad(elev_mask)
        self.sat_calc = SatellitePositionCalculator()
        self._ion_gps = (
            np.asarray(ion_gps, dtype=float).reshape(8).copy()
            if ion_gps is not None and len(np.asarray(ion_gps).reshape(-1)) >= 8
            else None
        )
        self._ion_bds = (
            np.asarray(ion_bds, dtype=float).reshape(8).copy()
            if ion_bds is not None and len(np.asarray(ion_bds).reshape(-1)) >= 8
            else None
        )
        # Bias-SINEX DCB 属外部产品：仅当 SPP_USE_BSX_DCB=1 且路径存在时加载
        self._dcb = None
        if os.environ.get("SPP_USE_BSX_DCB", "").strip().lower() in ("1", "true", "yes", "on"):
            default_bsx = os.path.normpath(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..",
                    "..",
                    "..",
                    "compass-master",
                    "data",
                    "2021",
                    "196",
                    "products",
                    "cas",
                    "CAS0MGXRAP_20211960000_01D_01D_DCB.BSX",
                )
            )
            if os.path.isfile(default_bsx):
                try:
                    self._dcb = BiasSinexDCB.from_file(default_bsx)
                except Exception:
                    self._dcb = None
        # 最近一次迭代的调试信息（用于排查SPP失败原因）
        self.last_debug = {}
        self.last_x = None
        self.last_used_sats = None
        self.last_residuals = None
        self._nav_index_cache = {}
        self._sat_state_cache = {}
        # GREAT-PVT 对 BDS GEO 的常见处理：
        # - 轨道模型支持 GEO，但在部分流程会直接排除 C01~C05
        # - GEO 观测通常降权（观测噪声放大）
        # 在本数据集中 C59/C60 也表现为强异常，默认一并剔除（它们在 GREAT 也被标记为 GEO）。
        self.exclude_bds_prns = {1, 2, 3, 4, 5, 59, 60}
        self.downweight_bds_geo = True
        # GREAT-PVT: BDS2 IGSO/MEO code bias correction (Wanninger & Beer)
        self.correct_bds2_igso_meo_code_bias = True
        # 默认对齐 GREAT-PVT/LibGnut：多系统各 ISB + GLO 每星 IFB（见 _build_equations）。
        # SPP_RTKLIB_PNTPOS_ISB=1：强制全程 RTKLIB 式单参数 GLO（不尝试按星 IFB）。
        # SPP_GLO_ISB_MODE=great|rtklib|auto：未设 SPP_RTKLIB_PNTPOS_ISB 时选用；
        #   默认 great（与改前一致）；auto 在 R 星存在且按星 IFB 失败后再试 RTKLIB 单参数
        #   （部分数据上 RTKLIB 可能收敛但偏差很大，仅作可选实验）。
        self.rtklib_pntpos_isb = os.environ.get("SPP_RTKLIB_PNTPOS_ISB", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        _glo_m = os.environ.get("SPP_GLO_ISB_MODE", "auto").strip().lower()
        if self.rtklib_pntpos_isb:
            self._glo_isb_mode = "rtklib"
        elif _glo_m in ("rtklib", "pntpos", "rtk"):
            self._glo_isb_mode = "rtklib"
        elif _glo_m == "great":
            self._glo_isb_mode = "great"
        else:
            self._glo_isb_mode = "auto"
        # 单历元求解时由 solve() 写入：本历元状态维、GLO 卫星号 -> 列号（>= NX_BASE）
        self._solve_nx: int = self.NX_BASE
        self._glo_ifb_col: dict = {}
        if glo_ifb_ema_beta is not None:
            self._glo_ifb_ema_beta = float(max(0.0, min(1.0, glo_ifb_ema_beta)))
        else:
            raw = os.environ.get("SPP_GLO_EMA", "0.12").strip()
            if not raw:
                raw = "0.12"
            try:
                self._glo_ifb_ema_beta = float(max(0.0, min(1.0, float(raw))))
            except ValueError:
                self._glo_ifb_ema_beta = 0.12
        # GREAT 式 GLO_ISB(x[4])、按星 IFB 的跨历元 EMA（仅成功历元更新）
        self._glo_isb_ema: Optional[float] = None
        self._glo_ifb_ema: dict[int, float] = {}
        # RTKLIB 单参 GLO 时单独记忆 x[4]，避免与 GREAT 语义混用
        self._glo_rtk_isb_ema: Optional[float] = None
        # solve 内某次 WLS 尝试是否为 RTKLIB 式（含 auto 第二遍）；供 _build_equations 与 rtklib_pntpos_isb 组合
        self._use_rtklib_glo_this_epoch: bool = False
        try:
            self._glo_pr_noise_fact = float(os.environ.get("SPP_GLO_PR_FACT", "2.2"))
        except ValueError:
            self._glo_pr_noise_fact = 2.2
        if self._glo_pr_noise_fact < 0.5:
            self._glo_pr_noise_fact = 0.5

    def _equation_use_rtklib_glo(self) -> bool:
        return bool(self.rtklib_pntpos_isb or self._use_rtklib_glo_this_epoch)

    def reset_glo_bias_memory(self) -> None:
        """换观测段/重跑前清空 GLONASS 偏差记忆。"""
        self._glo_isb_ema = None
        self._glo_ifb_ema.clear()
        self._glo_rtk_isb_ema = None

    def _apply_glo_ema_init(
        self,
        x: np.ndarray,
        glo_ids: List[int],
        use_rtklib_glo: bool,
    ) -> None:
        b = self._glo_ifb_ema_beta
        if b <= 0:
            return
        if use_rtklib_glo:
            if self._glo_rtk_isb_ema is not None and len(x) > 4:
                x[4] = float(self._glo_rtk_isb_ema)
            return
        if not glo_ids:
            return
        if self._glo_isb_ema is not None and len(x) > 4:
            x[4] = float(self._glo_isb_ema)
        for sid, col in self._glo_ifb_col.items():
            if col < len(x) and sid in self._glo_ifb_ema:
                x[col] = float(self._glo_ifb_ema[sid])

    def _update_glo_ema_after_success(
        self,
        x: np.ndarray,
        glo_ids: List[int],
        use_rtklib_glo: bool,
    ) -> None:
        b = self._glo_ifb_ema_beta
        if b <= 0:
            return
        if use_rtklib_glo:
            if glo_ids and len(x) > 4:
                v4 = float(x[4])
                if self._glo_rtk_isb_ema is None:
                    self._glo_rtk_isb_ema = v4
                else:
                    self._glo_rtk_isb_ema = (1.0 - b) * self._glo_rtk_isb_ema + b * v4
            return
        if not glo_ids or len(x) < self.NX_BASE:
            return
        v4 = float(x[4])
        if self._glo_isb_ema is None:
            self._glo_isb_ema = v4
        else:
            self._glo_isb_ema = (1.0 - b) * self._glo_isb_ema + b * v4
        for sid, col in self._glo_ifb_col.items():
            if col >= len(x):
                continue
            vc = float(x[col])
            if sid not in self._glo_ifb_ema:
                self._glo_ifb_ema[sid] = vc
            else:
                self._glo_ifb_ema[sid] = (1.0 - b) * self._glo_ifb_ema[sid] + b * vc

    @staticmethod
    def _bds2_igso_meo_code_bias_m(sat_id: int, elev_deg: float, code: Optional[str]) -> float:
        """GREAT-PVT gqualitycontrol.cpp: apply_IGSO_MEO() 的最小复刻（单位：m）。

        适用对象：
        - BDS-2: C06..C16（GREAT: sat <= C05 or sat > C16 跳过）
        - sat_type: C11/C12/C14 视为 MEO，其余视为 IGSO
        - 频段: B1/B2/B3 -> 用观测码的 band 2/7/6 映射
        """
        if sat_id <= 5 or sat_id > 16:
            return 0.0
        if not np.isfinite(elev_deg):
            return 0.0
        c = (code or "").upper()
        if len(c) < 2:
            return 0.0

        # band mapping for BDS2
        # - C2*: B1I  -> BAND_2
        # - C7*: B2I  -> BAND_7
        # - C6*: B3I  -> BAND_6
        band = None
        if c.startswith("C2"):
            band = "B1"
        elif c.startswith("C7"):
            band = "B2"
        elif c.startswith("C6"):
            band = "B3"
        else:
            return 0.0

        sat_type = "MEO" if sat_id in (11, 12, 14) else "IGSO"

        # GREAT tables: (index 0..9) for elevation bins of 10 deg, linear interp
        IGSO = {
            "B1": [-0.55, -0.40, -0.34, -0.23, -0.15, -0.04, 0.09, 0.19, 0.27, 0.35],
            "B2": [-0.71, -0.36, -0.33, -0.19, -0.14, -0.03, 0.08, 0.17, 0.24, 0.33],
            "B3": [-0.27, -0.23, -0.21, -0.15, -0.11, -0.04, 0.05, 0.14, 0.19, 0.32],
        }
        MEO = {
            "B1": [-0.47, -0.38, -0.32, -0.23, -0.11, 0.06, 0.34, 0.69, 0.97, 1.05],
            "B2": [-0.40, -0.31, -0.26, -0.18, -0.06, 0.09, 0.28, 0.48, 0.64, 0.69],
            "B3": [-0.22, -0.15, -0.13, -0.10, -0.04, 0.05, 0.14, 0.27, 0.36, 0.47],
        }
        tbl = IGSO if sat_type == "IGSO" else MEO
        arr = tbl[band]

        e = float(elev_deg)
        x = e / 10.0
        i0 = int(np.floor(x))
        if i0 < 0:
            return float(arr[0])
        if i0 >= 9:
            return float(arr[9])
        t = x - i0
        return float(arr[i0] * (1.0 - t) + arr[i0 + 1] * t)
        
    def _solve_wls_epoch(
        self,
        obs: GNSSRawObservation,
        nav_list: List[NavRecord],
        approx_pos: Optional[np.ndarray],
        glo_ids: List[int],
        use_rtklib_glo: bool,
    ) -> Tuple[np.ndarray, float, int]:
        """单历元迭代 WLS；use_rtklib_glo=True 时不扩 GLO 每星 IFB。"""
        self._use_rtklib_glo_this_epoch = use_rtklib_glo
        self.last_used_sats = None
        self.last_residuals = None
        if use_rtklib_glo:
            self._solve_nx = self.NX_BASE
            self._glo_ifb_col = {}
        else:
            self._solve_nx = self.NX_BASE + len(glo_ids)
            self._glo_ifb_col = {
                sid: self.NX_BASE + k for k, sid in enumerate(glo_ids)
            }

        x = np.zeros(self._solve_nx, dtype=float)
        if approx_pos is not None:
            x[:3] = approx_pos
        self._apply_glo_ema_init(x, glo_ids, use_rtklib_glo)
        BB0 = self._build_bancroft_bb_gps(obs, nav_list)
        if BB0 is not None:
            sol4 = bancroft_solve_lorentz(BB0)
            if sol4 is not None and np.all(np.isfinite(sol4)):
                x[3] = float(sol4[3])

        exc = set()
        max_retries = 3
        retry = 0
        converged = False
        iter_count = 0

        while retry <= max_retries:
            converged = False
            for iter_count in range(self.MAXITR):
                H, v, R, sat_info, used_sats = self._build_equations(
                    x, obs, nav_list, iter_count, exc=exc
                )
                if H is None or v is None or R is None or len(v) < 4:
                    self.last_debug = {**self.last_debug, 'iter': iter_count, 'fail_reason': 'H_none_or_v_lt4'}
                    self.last_x = x.copy()
                    return x[:3], x[3], 0

                try:
                    w = 1.0 / np.diag(R)
                    Hw = H.T * w
                    N = Hw @ H
                    y = Hw @ v
                    if np.linalg.cond(N) > 1e12:
                        self.last_debug = {**self.last_debug, 'iter': iter_count, 'fail_reason': 'ill_conditioned'}
                        self.last_x = x.copy()
                        return x[:3], x[3], 0
                    dx = np.linalg.solve(N, y)
                except np.linalg.LinAlgError:
                    self.last_debug = {**self.last_debug, 'iter': iter_count, 'fail_reason': 'linalg_error'}
                    self.last_x = x.copy()
                    return x[:3], x[3], 0

                if np.any(~np.isfinite(dx)):
                    self.last_debug = {**self.last_debug, 'iter': iter_count, 'fail_reason': 'bad_solution'}
                    self.last_x = x.copy()
                    return x[:3], x[3], 0

                # GLONASS 按星 IFB 时前几步钟差/坐标耦合大，过严门限易误杀；略放宽（仍仅单历元 SPP）
                loose_glo = bool(glo_ids) and not use_rtklib_glo
                lim_pos = 280.0 if loose_glo else self.MAX_DX_POS
                lim_clk = 2.5e5 if loose_glo else self.MAX_DX_CLK
                if (iter_count > 0 or approx_pos is None) and (
                    np.linalg.norm(dx[:3]) > lim_pos or abs(dx[3]) > lim_clk
                ):
                    self.last_debug = {**self.last_debug, 'iter': iter_count, 'fail_reason': 'dx_too_large'}
                    self.last_x = x.copy()
                    return x[:3], x[3], 0

                x = x + dx

                if np.linalg.norm(dx[:3]) < 1e-4 and abs(dx[3]) < 1e-3:
                    converged = True
                    self.last_used_sats = used_sats
                    self.last_residuals = v.copy()
                    bad = []
                    for j, sat_key in enumerate(used_sats):
                        if sat_key is None:
                            continue
                        if abs(v[j]) > self.RES_EXC_TH:
                            bad.append(sat_key)
                    if bad and retry < max_retries:
                        exc.update(bad)
                        retry += 1
                        break
                    break

            if converged:
                break
            break

        if not converged or len(getattr(self, "last_used_sats", []) or []) < 4:
            self.last_debug = {**self.last_debug, 'fail_reason': 'not_converged_or_few_sats', 'iter': iter_count}
            self.last_x = x.copy()
            return x[:3], x[3], 0

        self.last_x = x.copy()
        self._update_glo_ema_after_success(x, glo_ids, use_rtklib_glo)
        return x[:3], x[3], 1

    def solve(
        self,
        obs: GNSSRawObservation,
        nav_list: List[NavRecord],
        approx_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float, int]:
        """
        SPP解算

        Args:
            obs: 观测数据（一个历元）
            nav_list: 导航星历列表
            approx_pos: 概略位置 ECEF [x, y, z] (m)。为 None 时，若 obs 带有 RINEX 头
            APPROX POSITION XYZ（由读取器写入 approx_position），则用之；否则用地心 (0,0,0)。

        Returns:
            (position, clock_bias, status)
            - position: 位置 ECEF [x, y, z] (m)
            - clock_bias: 接收机钟差 (m)
            - status: 解算状态 (0-失败, 1-单点定位)

        环境变量 SPP_GLO_ISB_MODE（未设 SPP_RTKLIB_PNTPOS_ISB 时）：
            great（默认）仅每星 IFB。
            auto         有 R 星时先每星 IFB，失败再试 RTKLIB 单参数（可能假收敛、大偏差）。
            rtklib       仅 RTKLIB 单参数。
        """
        glo_ids = sorted({o.sat_id for o in obs.observations if o.system == "R"})
        observed_systems = {o.system for o in obs.observations if o.system in "GRECIJ"}
        self._r_only_this_epoch = observed_systems == {"R"}

        if approx_pos is None:
            ap_hdr = getattr(obs, "approx_position", None)
            if ap_hdr is not None:
                ap_h = np.asarray(ap_hdr, dtype=float).reshape(3)
                if np.all(np.isfinite(ap_h)) and float(np.linalg.norm(ap_h)) > 100.0:
                    approx_pos = ap_h

        # Each attempt is (observation epoch, GLO ids, use single GLO ISB, label).
        # In auto mode preserve the precise per-satellite model when it works;
        # if it becomes ill-conditioned, exclude GLO before trying the lower-
        # accuracy single-ISB model.
        if self._r_only_this_epoch:
            attempts = [(obs, glo_ids, True, "r_only")]
        elif self._glo_isb_mode == "rtklib":
            attempts = [(obs, glo_ids, True, "rtklib")]
        elif self._glo_isb_mode == "great":
            attempts = [(obs, glo_ids, False, "great")]
        else:
            attempts = [(obs, glo_ids, False, "great")]
            if glo_ids:
                non_glo = [o for o in obs.observations if o.system != "R"]
                if len(non_glo) >= 4:
                    attempts.append((replace(obs, observations=non_glo), [], False, "exclude_glo"))
                attempts.append((obs, glo_ids, True, "rtklib"))

        last: Optional[Tuple[np.ndarray, float, int]] = None
        last_label = "none"
        for attempt_obs, attempt_glo_ids, use_rtk, label in attempts:
            last_label = label
            self.last_debug = {"glo_isb_attempt": label}
            last = self._solve_wls_epoch(
                attempt_obs, nav_list, approx_pos, attempt_glo_ids, use_rtk
            )
            if last[2] == 1:
                self.last_debug["glo_isb_used"] = label
                return last

        assert last is not None
        self.last_debug["glo_isb_used"] = last_label
        return last

    def estimate_rover_from_base_sd(
        self,
        base_pos: np.ndarray,
        obs_rover: GNSSRawObservation,
        obs_base: GNSSRawObservation,
        nav_list: List[NavRecord],
        min_sats: int = 5,
    ) -> Optional[np.ndarray]:
        """
        已知基站 ECEF 时，用 GPS L1 伪距单差 + 线性化几何距离估计流动站概略位置。

        当 RINEX 头 APPROX 与基站几乎重合（占位/错误）而真实基线为 km 级时，绝对 SPP
        可能在基站附近落入错误不动点；本方法给出与基站同一历元、同一钟差差分模型下的粗解，
        作为 SPP 初值（对齐「相对定位几何」直觉）。
        """
        base_pos = np.asarray(base_pos, dtype=float).reshape(3)
        obs_gps_time = obs_rover.week * 604800.0 + obs_rover.timestamp
        rows: List[List[float]] = []
        y: List[float] = []
        for sat_obs in obs_rover.observations:
            if sat_obs.system != "G":
                continue
            o_b = obs_base.get_satellite_by_id(sat_obs.sat_id, sat_obs.system)
            if o_b is None:
                continue
            pr = float(self._get_pseudorange(sat_obs))
            pb = float(self._get_pseudorange(o_b))
            if pr <= 0.0 or pb <= 0.0 or not np.isfinite(pr) or not np.isfinite(pb):
                continue
            nav = self._find_nav(sat_obs.sat_id, sat_obs.system, obs_gps_time, nav_list)
            if nav is None:
                continue
            transmit_time_rough = obs_gps_time - pb / CLIGHT
            _, _, sat_clk_rough = self._compute_sat_pos(nav, transmit_time_rough)
            if sat_clk_rough is None:
                sat_clk_rough = 0.0
            transmit_time = transmit_time_rough - sat_clk_rough
            sat_pos, _, _ = self._compute_sat_pos(nav, transmit_time)
            if sat_pos is None:
                continue
            dx = sat_pos - base_pos
            R = float(np.linalg.norm(dx))
            if R < 1.0e6:
                continue
            los = dx / R
            user_llh = ecef2llh(base_pos)
            azel = self._compute_azel(sat_pos, base_pos, user_llh)
            if azel[1] < self.elev_mask:
                continue
            sd = pr - pb
            rows.append([-los[0], -los[1], -los[2], 1.0])
            y.append(sd)
        if len(y) < min_sats:
            return None
        H = np.asarray(rows, dtype=float)
        yv = np.asarray(y, dtype=float)
        try:
            sol, *_ = np.linalg.lstsq(H, yv, rcond=None)
        except np.linalg.LinAlgError:
            return None
        dr = sol[:3]
        if not np.all(np.isfinite(dr)) or np.linalg.norm(dr) > 5.0e6:
            return None
        r = base_pos + dr
        if np.linalg.norm(r) < 1000.0:
            return None
        return r

    def _build_bancroft_bb_gps(
        self, obs: GNSSRawObservation, nav_list: List[NavRecord]
    ) -> Optional[np.ndarray]:
        """组装 GPS-only Bancroft 矩阵 BB(n,4)，列与 G-Nut gbancroft 输入一致。"""
        rows: List[List[float]] = []
        obs_gps_time = obs.week * 604800.0 + obs.timestamp
        for sat_obs in obs.observations:
            if sat_obs.system != "G":
                continue
            pr = float(self._get_pseudorange(sat_obs))
            if pr <= 0.0 or not np.isfinite(pr):
                continue
            nav = self._find_nav(sat_obs.sat_id, "G", obs_gps_time, nav_list)
            if nav is None:
                continue
            transmit_time_rough = obs_gps_time - pr / CLIGHT
            _, _, sat_clk_rough = self._compute_sat_pos(nav, transmit_time_rough)
            if sat_clk_rough is None:
                sat_clk_rough = 0.0
            transmit_time = transmit_time_rough - sat_clk_rough
            sat_pos, _, sat_clk = self._compute_sat_pos(nav, transmit_time)
            if sat_pos is None or sat_clk is None:
                continue
            p4 = pr + CLIGHT * float(sat_clk)
            rows.append([float(sat_pos[0]), float(sat_pos[1]), float(sat_pos[2]), p4])
        if len(rows) < 4:
            return None
        return np.asarray(rows, dtype=float)

    def estimate_bancroft_gps_only(
        self, obs: GNSSRawObservation, nav_list: List[NavRecord]
    ) -> Optional[np.ndarray]:
        """
        仅用 GPS 伪距的 Bancroft 闭式近似位置（ECEF m），对齐 GREAT-PVT / G-Nut 首历元策略。
        多系统 SPP 前可用本结果作初值；失败返回 None。
        """
        BB = self._build_bancroft_bb_gps(obs, nav_list)
        if BB is None:
            return None
        sol4 = bancroft_solve_lorentz(BB)
        if sol4 is None or not np.all(np.isfinite(sol4)):
            return None
        xyz = sol4[:3]
        r = float(np.linalg.norm(xyz))
        # 地心距下界不可用 wgs84.a−δ：椭球面上多数纬度 r < a（两极附近约 b），误杀合法 Bancroft 初值。
        r_lo = float(wgs84.b) - 2.5e4
        r_hi = float(wgs84.a) + 5.0e5
        if r < r_lo or r > r_hi:
            return None
        return xyz.copy()

    def _build_equations(self,
                        x: np.ndarray,
                        obs: GNSSRawObservation,
                        nav_list: List[NavRecord],
                        iter_count: int = 0,
                        exc: Optional[set] = None) -> Tuple:
        """
        建立观测方程
        
        Returns:
            (H, v, R, sat_info, used_sats)；H 行为 nx 维（含 GREAT 风格 GLO 每星 IFB 列）。
        """
        nx = int(getattr(self, "_solve_nx", self.NX_BASE))
        glo_ifb_col: dict = getattr(self, "_glo_ifb_col", None) or {}

        user_pos = x[:3]
        # 与 pntpos.c 一致：主钟差默认取 GPS
        user_clk = x[3]  # GPS clock bias (m)
        
        # 计算GPS绝对时间（周 * 604800 + 周内秒）
        obs_gps_time = obs.week * 604800.0 + obs.timestamp
        
        if exc is None:
            exc = set()
        if x.shape[0] < nx:
            self.last_debug = {"fail_reason": "state_len_mismatch", "nx": nx, "len_x": int(x.shape[0])}
            return None, None, None, [], []

        H_list = []
        v_list = []
        var_list = []
        sat_info = []
        used_sats = []  # 与 v/H 同步的 (system, sat_id)，constraint 行为 None
        
        total_obs = len(obs.observations)
        nav_found = 0
        sat_pos_computed = 0
        elev_passed = 0
        
        # mask：标记哪些系统在本历元有观测（用于添加秩约束）
        mask = [0] * self.NUM_SYS

        for sat_obs in obs.observations:
            if (sat_obs.system, sat_obs.sat_id) in exc:
                continue
            # 只处理定位系统 (排除SBAS和IRNSS增强系统)
            if sat_obs.system not in ['G', 'E', 'C', 'R', 'J']:
                continue

            # GREAT-PVT 风格：默认剔除 BDS C01~C05（GEO）
            if sat_obs.system == "C" and sat_obs.sat_id in self.exclude_bds_prns:
                continue

            sys_idx = self._sys_index(sat_obs.system, sat_obs.sat_id)
            if sys_idx < 0:
                continue
            
            # 查找对应的导航星历
            nav = self._find_nav(sat_obs.sat_id, sat_obs.system, obs_gps_time, nav_list)
            if nav is None:
                continue
            nav_found += 1
            
            # 获取伪距观测值（优先L1/E1/B1）
            P1_raw_meas = self._get_pseudorange(sat_obs)
            if P1_raw_meas == 0.0 or np.isnan(P1_raw_meas):
                continue
            P1_raw = float(P1_raw_meas)

            # GREAT-PVT 风格 _applyDCB：对选用的码型应用卫星 DCB（BSX），将其统一到本卫星的参考码体系
            if self._dcb is not None:
                sat_prn = f"{sat_obs.system}{sat_obs.sat_id:02d}"
                if sat_obs.code_pr1:
                    b = self._dcb.sat_code_bias_m(sat_prn, sat_obs.code_pr1[0:3])
                    if b is not None:
                        P1_raw = float(P1_raw) - b
            
            # 计算信号发射时刻（参考COMPASS satposs）
            # **重要**：发射时刻用原始伪距，不应用TGD改正！
            # 因为TGD是系统性偏差，不影响传播时间
            # Step 1: 粗略发射时刻（用原始伪距）
            sat_state = self._cached_sat_state(sat_obs, obs_gps_time, nav_list, P1_raw_meas)
            if sat_state is None:
                continue
            nav, sat_pos, sat_vel, sat_clk = sat_state
            transmit_time_rough = obs_gps_time - P1_raw_meas / CLIGHT
            
            # 调试：打印第一颗GPS卫星的详细信息
            # Step 2: 计算卫星钟差（用粗略时刻）
            sat_clk_rough = sat_clk
            if sat_clk_rough is None:
                sat_clk_rough = 0.0
            
            # Step 3: 精确发射时刻（减去钟差）
            transmit_time = transmit_time_rough - sat_clk_rough
            
            # Step 4: 用精确时刻计算卫星位置和钟差
            sat_pos, sat_vel, sat_clk = sat_state[1], sat_state[2], sat_state[3]
            if sat_pos is None:
                continue
            sat_pos_computed += 1
            
            # 计算几何距离和视线向量（参考COMPASS geodist函数）
            # COMPASS使用简化的Sagnac改正：只改正距离，不旋转卫星位置
            dx = sat_pos - user_pos
            r_euclidean = np.linalg.norm(dx)  # 欧几里得距离
            los = dx / r_euclidean  # 视线单位向量（用欧几里得距离归一化）
            # Sagnac效应改正：r + OMGE*(rs[0]*rr[1]-rs[1]*rr[0])/CLIGHT
            sagnac = self.OMGE * (sat_pos[0] * user_pos[1] - sat_pos[1] * user_pos[0]) / CLIGHT
            r = r_euclidean + sagnac  # 几何距离（含Sagnac改正）
            
            # 计算高度角/方位角（需要用户位置）
            user_llh = None
            azel = None
            if np.linalg.norm(user_pos) > 1000:  # 用户位置有效（距离地心>1km）
                user_llh = ecef2llh(user_pos)
                azel = self._compute_azel(sat_pos, user_pos, user_llh)
                elev = azel[1]
            else:
                elev = np.pi / 2  # 初始未知时假设天顶，避免直接筛掉
            
            # 高度角截止
            if elev < self.elev_mask:
                continue
            elev_passed += 1
            
            # 应用TGD/BGD改正到观测伪距
            # 观测伪距：优先使用电离层无关组合（若 L2 存在且系统支持）
            # 对齐 pntpos.c 的思路：IFLC 时观测噪声方差需要放大约 9 倍
            P = None
            use_iflc = False
            # GREAT-PVT: BDS2 IGSO/MEO 码偏差经验改正（只改观测，不影响发射时刻）
            P1_raw = float(P1_raw_meas)
            if self.correct_bds2_igso_meo_code_bias and sat_obs.system == "C" and sat_obs.code_pr1:
                P1_raw += self._bds2_igso_meo_code_bias_m(
                    sat_obs.sat_id, np.rad2deg(elev), sat_obs.code_pr1
                )

            # PRIDE-PPPAR 风格“可配置/可降级”的最小版：
            # - BDS: 只有 slot1(B2I/B2b, C7*) 存在时才做 IFLC（用广播 TGD 的简化分支更可靠）
            # - 否则：BDS 退回单频（对齐 RTKLIB pntpos.c prange() 的 else 分支，不丢弃该星）
            allow_iflc = True
            if sat_obs.system == "C":
                c2 = (sat_obs.code_pr2 or "").upper()
                # 允许的 BDS 双频：B1+B2b(C7*) 或 B1+B3(C6*)
                # B2a(C5*) 等：不做 IFLC，仅用单频 + 广播改正（与 RTKLIB 单频一致）
                allow_iflc = c2.startswith("C7") or c2.startswith("C6")

            if (
                allow_iflc
                and sat_obs.pseudorange_L2 is not None
                and sat_obs.pseudorange_L2 != 0.0
                and np.isfinite(sat_obs.pseudorange_L2)
            ):
                # 注意：这里不对 P1/P2 直接扣同一个 tgd，而是用 _prange_iflc() 做系统特定修正（对齐 pntpos.c）
                P1 = float(P1_raw)
                P2 = float(sat_obs.pseudorange_L2)
                if self.correct_bds2_igso_meo_code_bias and sat_obs.system == "C" and sat_obs.code_pr2:
                    P2 += self._bds2_igso_meo_code_bias_m(
                        sat_obs.sat_id, np.rad2deg(elev), sat_obs.code_pr2
                    )
                if self._dcb is not None and sat_obs.system == "C" and sat_obs.code_pr2:
                    sat_prn = f"{sat_obs.system}{sat_obs.sat_id:02d}"
                    b2 = self._dcb.sat_code_bias_m(sat_prn, sat_obs.code_pr2[0:3])
                    if b2 is not None:
                        P2 = float(P2) - b2
                P_if = self._prange_iflc(sat_obs.system, P1, P2, nav, sat_obs)
                if P_if is not None and np.isfinite(P_if):
                    P = P_if
                    use_iflc = True
            if P is None:
                P = self._apply_code_bias(
                    P1_raw, nav, sat_obs.system, code_pr1=sat_obs.code_pr1
                )
            
            # GREAT-PVT gsppmodel::isbCorrection：GLO = GLO_ISB + GLO_IFB(按星)；其它系统仅 ISB
            isb = 0.0
            if self._equation_use_rtklib_glo():
                if sat_obs.system == "R":
                    isb = x[4]
            elif sat_obs.system == "R":
                gc = glo_ifb_col.get(sat_obs.sat_id)
                if gc is not None and gc < nx:
                    isb = x[4] + x[gc]
                else:
                    isb = x[4]
            elif sys_idx != 0:
                isb = x[3 + sys_idx]

            # GREAT-PVT 风格：BDS IFB 仅在与 IFLC 对应的第二码组合上使用；单频时不引入 IFB（对齐 RTKLIB 无额外 IFB）
            ifb = 0.0
            if sat_obs.system == "C" and use_iflc:
                c2 = (sat_obs.code_pr2 or "").upper()
                if c2.startswith("C6"):
                    ifb = x[self.IDX_IFB_C6]
                else:
                    ifb = x[self.IDX_IFB_C7]

            # 计算理论伪距
            # COMPASS公式: v = P - (r + cdtr - CLIGHT*dts + dion + dtrp)
            rho = r + user_clk - CLIGHT * sat_clk
            
            # 电离层延迟改正（Klobuchar模型）
            if use_iflc:
                dion = 0.0
            elif user_llh is not None and azel is not None:
                azel_iono = np.array([azel[0], elev])  # [az, el]
                dion, vion = ionocorr(
                    obs.timestamp,
                    self._ion_gps,
                    self._ion_bds,
                    user_llh,
                    azel_iono,
                    sat_obs.system,
                )
                # 频点缩放（对齐 pntpos.c: dion*=SQR(FREQ1/freq) 等）
                # 这里用近似频点代替 sat2freq() 的精确频率。
                f_ref = 1575.42e6
                f_obs = self._freq_L1_hz_for_obs(sat_obs, nav)
                if f_obs > 0.0:
                    scale = (f_ref / f_obs) ** 2
                    dion *= scale
                    vion *= scale
            else:
                dion = 0.0
            
            rho += dion
            
            # 对流层延迟改正（简化Saastamoinen模型）
            if user_llh is not None:
                trop_delay = self._tropospheric_delay(user_llh, elev)
            else:
                trop_delay = 0.0
            
            rho += trop_delay
            
            # 观测残差
            v = (P - rho) - isb - ifb
            
            # 设计矩阵：nx 维（含 GLO 每星 IFB 列）
            H = np.zeros(nx)
            H[0:3] = -los
            H[3] = 1.0
            if self._equation_use_rtklib_glo():
                if sat_obs.system == "R":
                    if getattr(self, "_r_only_this_epoch", False):
                        # R-only: x[3] is the receiver GLONASS clock. Constrain
                        # x[4] instead of estimating a collinear GPS-GLO ISB.
                        mask[0] = 1
                    else:
                        H[4] = 1.0
                        mask[0] = 1
                        mask[1] = 1
                else:
                    mask[0] = 1
            elif sat_obs.system == "R":
                H[4] = 1.0
                gc = glo_ifb_col.get(sat_obs.sat_id)
                if gc is not None and gc < nx:
                    H[gc] = 1.0
                mask[0] = 1
                mask[1] = 1
            elif sys_idx != 0:
                H[3 + sys_idx] = 1.0
                mask[sys_idx] = 1
            else:
                mask[0] = 1

            # IFB 偏导（仅 IFLC 时）
            if sat_obs.system == "C" and use_iflc:
                c2 = (sat_obs.code_pr2 or "").upper()
                if c2.startswith("C6"):
                    H[self.IDX_IFB_C6] = 1.0
                else:
                    H[self.IDX_IFB_C7] = 1.0
            
            # 观测噪声方差
            var = self._variance(elev, sat_obs.system)
            if use_iflc:
                var *= 9.0
            # GREAT: BDS GEO 观测降权（gbiasmodel.cpp: factor=5.0）
            if (
                self.downweight_bds_geo
                and sat_obs.system == "C"
                and (sat_obs.sat_id <= 5 or sat_obs.sat_id >= 59)
            ):
                var *= 25.0
            
            H_list.append(H)
            v_list.append(v)
            var_list.append(var)
            used_sats.append((sat_obs.system, sat_obs.sat_id))
            sat_info.append({
                'sat_id': sat_obs.sat_id,
                'system': sat_obs.system,
                'elev': np.rad2deg(elev),
                'azim': 0.0,
                'residual': v
            })
        
        if len(H_list) == 0:
            # print(f"  错误：没有有效观测值")
            # print(f"    找到星历: {nav_found}, 计算位置: {sat_pos_computed}, 高度角通过: {elev_passed}")
            # print(f"    截止角: {np.rad2deg(self.elev_mask):.1f}°")
            self.last_debug = {
                'iter': iter_count,
                'total_obs': total_obs,
                'nav_found': nav_found,
                'sat_pos_computed': sat_pos_computed,
                'elev_passed': elev_passed,
                'n_used': 0,
            }
            return None, None, None, [], []
        
        # constraint to avoid rank-deficient (对齐 pntpos.c: 对未出现系统加约束)
        for si in range(self.NUM_SYS):
            if mask[si]:
                continue
            Hc = np.zeros(nx)
            Hc[3 + si] = 1.0
            H_list.append(Hc)
            v_list.append(0.0)
            var_list.append(0.01)  # var=0.01 (m^2)
            used_sats.append(None)

        # IFB 正则项（GREAT-PVT 是滤波估计，参数天然有过程噪声约束；
        # 这里用每历元的小先验避免病态：ifb≈0 ± 100 m）
        for idx in (self.IDX_IFB_C7, self.IDX_IFB_C6):
            Hc = np.zeros(nx)
            Hc[idx] = 1.0
            H_list.append(Hc)
            v_list.append(0.0)
            var_list.append(100.0 ** 2)
            used_sats.append(None)

        # GLO 每星 IFB 先验（吸收 FDMA 码/信道残差）；已有 EMA 的星略收紧方差，利于稳定
        if not self._equation_use_rtklib_glo() and glo_ifb_col:
            for sid, col in sorted(glo_ifb_col.items(), key=lambda t: t[1]):
                Hc = np.zeros(nx)
                Hc[col] = 1.0
                H_list.append(Hc)
                v_list.append(0.0)
                if self._glo_ifb_ema_beta > 0.0 and sid in self._glo_ifb_ema:
                    var_list.append(55.0 ** 2)
                else:
                    var_list.append(100.0 ** 2)
                used_sats.append(None)

        H = np.array(H_list, dtype=float)
        v = np.array(v_list, dtype=float)
        R = np.diag(np.array(var_list, dtype=float))
        
        # 调试：打印第一次迭代的H和v
        self.last_debug = {
            'iter': iter_count,
            'total_obs': total_obs,
            'nav_found': nav_found,
            'sat_pos_computed': sat_pos_computed,
            'elev_passed': elev_passed,
            'n_used': len(H_list),
        }
        return H, v, R, sat_info, used_sats
    
    def _find_nav(self, sat_id: int, system: str, time: float, 
                  nav_list: List[NavRecord]) -> Optional[NavRecord]:
        """查找最近的导航星历"""
        cache_key = id(nav_list)
        cached = self._nav_index_cache.get(cache_key)
        if cached is None or cached[0] is not nav_list:
            name_to_sys = {
                'GPS': 'G',
                'GLONASS': 'R',
                'Galileo': 'E',
                'BDS': 'C',
                'QZSS': 'J',
                'IRNSS': 'I',
            }
            grouped = {}
            for order, nav in enumerate(nav_list):
                if isinstance(nav, GlonassEphemeris):
                    key = ('R', nav.sat_id)
                else:
                    key = (name_to_sys.get(nav.system, nav.system), nav.sat_id)
                grouped.setdefault(key, []).append((nav.toe, order, nav))
            index = {}
            for key, records in grouped.items():
                records.sort(key=lambda item: item[0])
                index[key] = ([item[0] for item in records], records)
            cached = (nav_list, index)
            self._nav_index_cache[cache_key] = cached

        entry = cached[1].get((system, sat_id))
        if entry is None:
            return None
        toes, records = entry
        pos = bisect_left(toes, time)
        candidates = []
        for i in (pos - 1, pos):
            if 0 <= i < len(records):
                candidates.append(records[i])
        if not candidates:
            return None
        best_toe, _, best_nav = min(candidates, key=lambda item: (abs(time - item[0]), item[1]))
        return best_nav if abs(time - best_toe) < 7200.0 else None

    def _cached_sat_state(self, sat_obs: SatelliteObservation, obs_gps_time: float, nav_list: List[NavRecord], p1_raw_meas: float):
        key = (id(nav_list), id(sat_obs), obs_gps_time, float(p1_raw_meas))
        cached = self._sat_state_cache.get(key)
        if cached is not None:
            return cached
        nav = self._find_nav(sat_obs.sat_id, sat_obs.system, obs_gps_time, nav_list)
        if nav is None:
            return None
        transmit_time_rough = obs_gps_time - p1_raw_meas / CLIGHT
        _, _, sat_clk_rough = self._compute_sat_pos(nav, transmit_time_rough)
        if sat_clk_rough is None:
            sat_clk_rough = 0.0
        transmit_time = transmit_time_rough - sat_clk_rough
        sat_pos, sat_vel, sat_clk = self._compute_sat_pos(nav, transmit_time)
        if sat_pos is None:
            return None
        cached = (nav, sat_pos, sat_vel, sat_clk)
        self._sat_state_cache[key] = cached
        return cached
    
    def _compute_sat_pos(self, nav: NavRecord, time: float) -> Tuple:
        """计算卫星位置"""
        # 使用通用函数支持多系统（含 GLONASS 广播星历）
        return self.sat_calc.compute_satellite_position(nav, time, nav.system)
    
    def _freq_L1_hz_for_obs(
        self, sat_obs: SatelliteObservation, nav: Optional[NavRecord] = None
    ) -> float:
        """单频电离层缩放用参考频率：BDS 按实际选用的 P1 码区分 B1I/B1C；GLO 按星历频率号。"""
        if sat_obs.system == "R" and isinstance(nav, GlonassEphemeris):
            return 1602.0e6 + nav.frq * 562500.0
        if sat_obs.system == "C" and sat_obs.code_pr1:
            fh = self._bds_rinex_code_to_freq_hz(sat_obs.code_pr1)
            if fh is not None and fh > 0.0:
                return fh
        return self._freq_L1_hz(sat_obs.system)

    def _get_pseudorange(self, sat_obs: SatelliteObservation) -> float:
        """
        获取伪距观测值（优先L1/E1/B1/G1频段）
        
        Args:
            sat_obs: 卫星观测数据
        
        Returns:
            伪距 (m)
        """
        # GPS: L1
        if sat_obs.system == 'G' and sat_obs.pseudorange_L1 > 0:
            return sat_obs.pseudorange_L1
        # Galileo: E1
        elif sat_obs.system == 'E' and sat_obs.pseudorange_L1 > 0:
            return sat_obs.pseudorange_L1
        # BDS: B1
        elif sat_obs.system == 'C' and sat_obs.pseudorange_L1 > 0:
            return sat_obs.pseudorange_L1
        # GLONASS: G1
        elif sat_obs.system == 'R' and sat_obs.pseudorange_L1 > 0:
            return sat_obs.pseudorange_L1
        # QZSS: L1 (与GPS相同)
        elif sat_obs.system == 'J' and sat_obs.pseudorange_L1 > 0:
            return sat_obs.pseudorange_L1
        
        return 0.0
    
    def _apply_code_bias(
        self,
        P_raw: float,
        nav: NavRecord,
        system: str,
        code_pr1: Optional[str] = None,
    ) -> float:
        """
        应用码偏差改正（TGD/BGD）
        
        参考: COMPASS prange()
        
        Args:
            P_raw: 原始伪距 (m)
            nav: 导航星历
            system: 卫星系统代码
            code_pr1: RINEX 选用的第一伪距码（BDS 时用于对齐 B1I/B1C 的广播改正）
        
        Returns:
            改正后的伪距 (m)
        """
        if system == 'G':  # GPS: TGD
            if not isinstance(nav, NavigationData):
                return P_raw
            return P_raw - nav.tgd * CLIGHT
        elif system == 'E':  # Galileo: BGD E5a/E1
            if not isinstance(nav, NavigationData):
                return P_raw
            return P_raw - nav.tgd * CLIGHT
        elif system == 'C':  # BDS: 借鉴 GREAT-PVT：tgd[0]=B1/B3, tgd[1]=B2/B3
            if not isinstance(nav, NavigationData):
                return P_raw
            c = (code_pr1 or "").upper()
            if c.startswith("C6"):  # B3
                return P_raw
            if c.startswith("C7"):  # B2I/B2b
                return P_raw - nav.tgd2 * CLIGHT
            # B1C/B1I
            return P_raw - nav.tgd * CLIGHT
        elif system == 'R':  # GLONASS: 不需要TGD改正
            return P_raw
        elif system == 'J':  # QZSS: TGD (与GPS相同)
            if not isinstance(nav, NavigationData):
                return P_raw
            return P_raw - nav.tgd * CLIGHT
        
        return P_raw
    
    def _compute_azel(self, sat_pos: np.ndarray, user_pos: np.ndarray, 
                     user_llh: np.ndarray) -> np.ndarray:
        """
        计算方位角和高度角
        
        参考: COMPASS src/LibGnss/rtkcmn.c - xyz2enu(), satazel()
        
        Args:
            sat_pos: 卫星位置 ECEF (m)
            user_pos: 用户位置 ECEF (m)
            user_llh: 用户位置 LLH [lat, lon, h] (rad, rad, m)
        
        Returns:
            [azimuth, elevation] (rad)
        """
        # LOS向量：用户→卫星（receiver-to-satellite unit vector）
        # 参考RTKLIB: e[i] = (rs[i] - rr[i]) / r
        los_ecef = sat_pos - user_pos
        los_ecef = los_ecef / np.linalg.norm(los_ecef)  # 归一化为单位向量
        
        # 转换到ENU坐标系
        lat, lon = user_llh[0], user_llh[1]
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        
        # ECEF到ENU的旋转矩阵（参考xyz2enu）
        # E[0]=-sinl;      E[3]=cosl;       E[6]=0.0;
        # E[1]=-sinp*cosl; E[4]=-sinp*sinl; E[7]=cosp;
        # E[2]=cosp*cosl;  E[5]=cosp*sinl;  E[8]=sinp;
        C_e2enu = np.array([
            [-sin_lon,         cos_lon,        0.0    ],
            [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
            [ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat]
        ])
        
        los_enu = C_e2enu @ los_ecef
        
        # 高度角和方位角（参考satazel）
        E, N, U = los_enu
        azimuth = np.arctan2(E, N) if np.dot(los_enu[:2], los_enu[:2]) > 1e-12 else 0.0
        elevation = np.arcsin(U)  # U是单位向量的z分量
        
        if azimuth < 0:
            azimuth += 2 * np.pi
        
        return np.array([azimuth, elevation])
    
    def _tropospheric_delay(self, llh: np.ndarray, elev: float) -> float:
        """
        对流层延迟（Saastamoinen模型）
        
        参考: COMPASS src/LibGnss/pntpos.c - tropmodel()
        """
        h = llh[2]  # 高度 (m)
        
        if elev <= 0:
            return 0.0
        
        # 限制高度范围，避免迭代过程中异常值
        h = np.clip(h, -1000, 10000)
        
        # 标准大气参数
        temp0 = 15.0  # 温度 (°C) at sea level
        pres0 = 1013.25  # 气压 (mbar) at sea level
        humi = 0.7  # 相对湿度
        
        # 高度改正
        temp = temp0 - 0.0065 * h
        pres = pres0 * (1.0 - 0.0000226 * h) ** 5.225
        
        # 检查数值有效性
        if not np.isfinite(pres) or pres <= 0:
            return 0.0
        
        # Saastamoinen模型
        e = humi * 6.108 * np.exp((17.15 * temp) / (234.7 + temp))
        
        # 天顶延迟
        z = np.pi / 2.0 - elev
        trph = 0.0022768 * pres / (1.0 - 0.00266 * np.cos(2.0 * llh[0]) - 0.00028 * h / 1000.0)
        trpw = 0.002277 * (1255.0 / (temp + 273.16) + 0.05) * e
        
        # 投影函数（简化）
        m = 1.001 / np.sqrt(0.002001 + np.sin(elev)**2)
        
        trop = (trph + trpw) * m
        
        # 最终检查
        if not np.isfinite(trop):
            return 0.0
        
        return trop
    
    def _variance(self, elev: float, system: str) -> float:
        """
        计算伪距观测噪声方差
        
        参考: COMPASS src/LibGnss/pntpos.c - varerr()
        RTKLIB默认: err[1]=0.3, err[2]=3.0 (不是0.3!)
        """
        # 基线噪声参数（对齐RTKLIB/COMPASS默认值）
        a = 0.3  # 基线噪声 (m)
        b = 3.0  # 高度角相关噪声 (m) - 修正！之前错误地设为0.3
        
        # 系统因子（传入为 RINEX 单字符或历史全名时均兼容）
        if system in ('G', 'GPS', 'J', 'QZSS'):
            fact = 1.0
        elif system in ('R', 'GLONASS'):
            # FDMA 码噪声通常大于 GPS；略提高因子可改善 G+R 单历元可收敛性（仍仅广播+观测）
            fact = float(self._glo_pr_noise_fact)
        elif system in ('C', 'BDS'):
            fact = 1.0
        elif system in ('E', 'Galileo'):
            fact = 1.0
        else:
            fact = 1.0
        
        sinel = np.sin(elev)
        var = fact**2 * (a**2 + b**2 / sinel**2)
        
        return var
