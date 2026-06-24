"""
RINEX Native Reader - Pure Python Implementation
基于COMPASS rinex.c的完整实现，不依赖第三方库
"""

import os
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta

from ..core.gnss_types import (
    GlonassEphemeris,
    GNSSRawObservation,
    NavigationData,
    NavRecord,
    SatelliteObservation,
)
from ..core.constants import CLIGHT

# COMPASS/RTKLIB obs code tables (from rtkcmn.c)
OBSCODES = [
    "", "1C", "1P", "1W", "1Y", "1M", "1N", "1S", "1L", "1E",
    "1A", "1B", "1X", "1Z", "2C", "2D", "2S", "2L", "2X", "2P",
    "2W", "2Y", "2M", "2N", "5I", "5Q", "5X", "7I", "7Q", "7X",
    "6A", "6B", "6C", "6X", "6Z", "6S", "6L", "8L", "8Q", "8X",
    "2I", "2Q", "6I", "6Q", "3I", "3Q", "3X", "1I", "1Q", "5A",
    "5B", "5C", "9A", "9B", "9C", "9X", "1D", "5D", "5P", "5Z",
    "6E", "7D", "7P", "7Z", "8D", "8P", "4A", "4B", "4X", ""
]

CODEPRIS = [
    ["CPYWMNSL", "PYWCMNDLSX", "IQX", "", "", "", ""],  # GPS
    ["PCABX", "PCABX", "IQX", "", "", "", ""],          # GLO
    ["CABXZ", "IQX", "IQX", "ABCXZ", "IQX", "", ""],   # GAL
    ["CLSXZ", "LSX", "IQXDPZ", "LSXEZ", "", "", ""],   # QZS
    ["C", "IQX", "", "", "", "", ""],                  # SBS
    ["IQXDPAN", "IQXDPZ", "DPX", "IQXA", "DPX", "", ""], # BDS
    ["ABCX", "ABCX", "", "", "", "", ""]               # IRN
]

SYS_GPS = 0
SYS_GLO = 1
SYS_GAL = 2
SYS_QZS = 3
SYS_SBS = 4
SYS_BDS = 5
SYS_IRN = 6


class RINEXNativeReader:
    """
    RINEX原生读取器（纯Python实现）
    
    参考: COMPASS src/LibGnss/rinex.c
    支持:
    - RINEX 2.x/3.x导航文件
    - RINEX 2.x/3.x观测文件
    """
    
    def __init__(self):
        """初始化"""
        self.version = 0.0
        self.file_type = ''
        self.sys = 0
        self.tsys = 0
        # 最近一次 read_nav 从文件头解析的广播电离层系数（8 元组 α0..β3），无则为 None
        self.nav_ion_gps: Optional[np.ndarray] = None
        self.nav_ion_bds: Optional[np.ndarray] = None
        self.obs_receiver_antenna = ""
        self.obs_antenna_delta_enu: Optional[np.ndarray] = None

    def read_nav(self, nav_file: str) -> List[NavRecord]:
        """
        读取RINEX导航文件
        
        Args:
            nav_file: 导航文件路径
        
        Returns:
            导航星历列表
        """
        if not os.path.exists(nav_file):
            raise FileNotFoundError(f"Nav file not found: {nav_file}")
        
        nav_list = []
        
        with open(nav_file, "r", encoding="utf-8", errors="replace") as fp:
            # 读取文件头
            ver, file_type, sys, tsys, ion_gps, ion_bds = self._read_header(fp)
            self.version = ver
            self.file_type = file_type
            self.sys = sys
            self.tsys = tsys
            self.nav_ion_gps = ion_gps
            self.nav_ion_bds = ion_bds
            
            print(f"RINEX Version: {ver:.2f}")
            print(f"File Type: {file_type}")
            print(f"System: {self._sys_name(sys)}")
            
            # 读取导航数据体
            while True:
                nav_data = self._read_nav_body(fp, ver, sys)
                if nav_data == 'EOF':
                    break  # 真正的EOF
                elif nav_data is None:
                    continue  # 不支持的系统，跳过并继续读取下一条
                else:
                    nav_list.append(nav_data)
        
        print(f"Loaded {len(nav_list)} navigation records")
        return nav_list
    
    @staticmethod
    def _parse_nav_ion_line(line: str) -> Optional[Tuple[str, Optional[Tuple[str, str]], Tuple[float, float, float, float]]]:
        """解析 RINEX3 导航头 IONOSPHERIC CORR 行。返回 (GPSA|GPSB|BDSA|BDSB, bds键或None, 四系数)。"""
        if len(line) < 61:
            return None
        label = line[60:].strip()
        if "IONOSPHERIC" not in label:
            return None
        parts = line[:60].split()
        if len(parts) < 5:
            return None
        tag = parts[0].upper()
        if tag not in ("GPSA", "GPSB", "BDSA", "BDSB"):
            return None
        try:
            v = tuple(float(parts[i]) for i in range(1, 5))
        except ValueError:
            return None
        bds_key: Optional[Tuple[str, str]] = None
        if tag.startswith("BDS") and len(parts) >= 7:
            bds_key = (parts[5], parts[6])
        return tag, bds_key, v

    def _read_header(self, fp) -> Tuple[float, str, int, int, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        读取RINEX导航文件头

        Returns:
            (version, file_type, sys, tsys, ion_gps_8, ion_bds_8)
        """
        ver = 0.0
        file_type = ""
        sys = 0  # SYS_GPS
        tsys = 0  # TSYS_GPS
        gps_a: Optional[Tuple[float, float, float, float]] = None
        gps_b: Optional[Tuple[float, float, float, float]] = None
        bds_blocks: Dict[Tuple[str, str], Dict[str, Tuple[float, float, float, float]]] = {}

        while True:
            line = fp.readline()
            if not line:
                break

            if len(line) < 60:
                continue

            label = line[60:].strip()

            # RINEX VERSION / TYPE
            if "RINEX VERSION" in label:
                ver = float(line[0:9].strip())
                file_type = line[20:21].strip()
                if ver >= 3.0 and len(line) > 40:
                    sys_char = line[40:41].strip()
                    sys = self._char_to_sys(sys_char)

            # TIME SYSTEM ID
            elif "TIME SYSTEM ID" in label:
                tsys_str = line[0:3].strip()
                if tsys_str == "GPS":
                    tsys = 0  # TSYS_GPS
                elif tsys_str in ("GLO", "UTC"):
                    tsys = 1  # TSYS_UTC
                elif tsys_str == "GAL":
                    tsys = 2  # TSYS_GAL
                elif tsys_str == "BDT":
                    tsys = 4  # TSYS_BDT

            elif "END OF HEADER" in label:
                break

            else:
                parsed = self._parse_nav_ion_line(line)
                if parsed is None:
                    continue
                tag, bkey, coef = parsed
                if tag == "GPSA":
                    gps_a = coef
                elif tag == "GPSB":
                    gps_b = coef
                elif tag == "BDSA" and bkey is not None:
                    bds_blocks.setdefault(bkey, {})["alpha"] = coef
                elif tag == "BDSB" and bkey is not None:
                    bds_blocks.setdefault(bkey, {})["beta"] = coef

        ion_gps: Optional[np.ndarray] = None
        if gps_a is not None and gps_b is not None:
            ion_gps = np.array(list(gps_a) + list(gps_b), dtype=float)

        ion_bds: Optional[np.ndarray] = None
        chosen: Optional[Tuple[str, str]] = None
        for k in bds_blocks:
            d = bds_blocks[k]
            if "alpha" in d and "beta" in d and k[0] == "C":
                chosen = k
                break
        if chosen is None:
            for k in bds_blocks:
                d = bds_blocks[k]
                if "alpha" in d and "beta" in d:
                    chosen = k
                    break
        if chosen is not None:
            d = bds_blocks[chosen]
            ion_bds = np.array(list(d["alpha"]) + list(d["beta"]), dtype=float)

        return ver, file_type, sys, tsys, ion_gps, ion_bds

    def _obs2code(self, obs: str) -> int:
        for i in range(1, len(OBSCODES)):
            if OBSCODES[i] == obs:
                return i
        return 0

    def _code2idx(self, sys: int, code: int) -> int:
        if code <= 0 or code >= len(OBSCODES):
            return -1
        obs = OBSCODES[code]
        band = obs[0]
        if sys == SYS_GPS:
            return {"1": 0, "2": 1, "5": 2}.get(band, -1)
        if sys == SYS_GLO:
            return {"1": 0, "2": 1, "3": 2, "4": 0, "6": 1}.get(band, -1)
        if sys == SYS_GAL:
            return {"1": 0, "7": 1, "5": 2, "6": 3, "8": 4}.get(band, -1)
        if sys == SYS_QZS:
            return {"1": 0, "2": 1, "5": 2, "6": 3}.get(band, -1)
        if sys == SYS_SBS:
            return {"1": 0, "5": 1}.get(band, -1)
        if sys == SYS_BDS:
            # 对齐 rtkcmn.c: code2freq_BDS() / code2idx()
            # 频点索引 (0..4):
            # 0: B1 (B1C/B1I)   band '1'/'2'
            # 1: B2 (B2I/B2b)   band '7'
            # 2: B2a            band '5'
            # 3: B3             band '6'
            # 4: B2ab           band '8'
            return {"1": 0, "2": 0, "7": 1, "5": 2, "6": 3, "8": 4}.get(band, -1)
        if sys == SYS_IRN:
            return {"5": 0, "9": 1}.get(band, -1)
        return -1

    def _getcodepri(self, sys: int, code: int) -> int:
        if code <= 0 or code >= len(OBSCODES):
            return 0
        j = self._code2idx(sys, code)
        if j < 0:
            return 0
        obs = OBSCODES[code]
        p = CODEPRIS[sys][j].find(obs[1])
        return 14 - p if p >= 0 else 0

    def _sys_char_to_idx(self, system: str) -> int:
        return {
            'G': SYS_GPS,
            'R': SYS_GLO,
            'E': SYS_GAL,
            'J': SYS_QZS,
            'S': SYS_SBS,
            'C': SYS_BDS,
            'I': SYS_IRN,
        }.get(system, SYS_GPS)

    def _select_obs(self, obs_values: dict, system: str, prefix: str, freq_idx: int) -> Tuple[float, int, float]:
        sys_idx = self._sys_char_to_idx(system)
        best_val = 0.0
        best_lli = 0
        best_snr = 0.0
        best_pri = -1
        for obs_type, (val, lli, snr) in obs_values.items():
            if not obs_type.startswith(prefix):
                continue
            code = self._obs2code(obs_type[1:3])
            if code == 0:
                continue
            if self._code2idx(sys_idx, code) != freq_idx:
                continue
            pri = self._getcodepri(sys_idx, code)
            if pri > best_pri:
                best_pri = pri
                best_val = val
                best_lli = lli
                best_snr = snr
        return best_val, best_lli, best_snr

    def _select_obs_with_code(
        self, obs_values: dict, system: str, prefix: str, freq_idx: int
    ) -> Tuple[float, int, float, Optional[str]]:
        """与 _select_obs() 相同，但返回被选中的观测类型字符串（如 'C2I'）。"""
        sys_idx = self._sys_char_to_idx(system)
        best_val = 0.0
        best_lli = 0
        best_snr = 0.0
        best_pri = -1
        best_type: Optional[str] = None
        for obs_type, (val, lli, snr) in obs_values.items():
            if not obs_type.startswith(prefix):
                continue
            code = self._obs2code(obs_type[1:3])
            if code == 0:
                continue
            if self._code2idx(sys_idx, code) != freq_idx:
                continue
            pri = self._getcodepri(sys_idx, code)
            if pri > best_pri:
                best_pri = pri
                best_val = val
                best_lli = lli
                best_snr = snr
                best_type = obs_type
        return best_val, best_lli, best_snr, best_type

    def _pick_bds_pseudorange_pair(
        self, obs_values: dict, sat_num: int
    ) -> Tuple[float, int, float, Optional[str], float, int, float, Optional[str]]:
        """BDS：按频点/代数显式选 P1/P2，避免 C2I/C7I/C6I/C1X/C5X 混排时 _select_obs 选错槽位。

        与 RTKLIB/pntpos 常用组合一致：BD2 优先 B1I+B2I；BD3 优先 B1C+B2a，其次 B1I+B2I 等。
        """
        bd3 = sat_num > 18

        def _get(code: str) -> Optional[Tuple[float, int, float]]:
            if code not in obs_values:
                return None
            val, lli, snr = obs_values[code]
            if val is None or val <= 0.0 or not np.isfinite(val):
                return None
            return float(val), int(lli), float(snr)

        if bd3:
            p1_order = ("C1X", "C1C", "C1P", "C1D", "C2I", "C2X", "C2Q", "C2P")
            p2_order = ("C5X", "C5Q", "C5D", "C5P", "C7I", "C7X", "C7Q", "C6I", "C6X", "C6Q")
        else:
            p1_order = ("C2I", "C2X", "C2Q", "C2P", "C1X", "C1C", "C1P", "C1D")
            p2_order = ("C7I", "C7X", "C7Q", "C5X", "C5Q", "C6I", "C6X", "C6Q")

        pseudo_l1, lli_l1, snr_l1, c1 = 0.0, 0, 0.0, None
        pseudo_l2, lli_l2, snr_l2, c2 = 0.0, 0, 0.0, None
        for c in p1_order:
            r = _get(c)
            if r is not None:
                pseudo_l1, lli_l1, snr_l1 = r
                c1 = c
                break
        for c in p2_order:
            if c == c1:
                continue
            r = _get(c)
            if r is not None:
                pseudo_l2, lli_l2, snr_l2 = r
                c2 = c
                break
        return pseudo_l1, lli_l1, snr_l1, c1, pseudo_l2, lli_l2, snr_l2, c2

    def _read_nav_body(
        self, fp, ver: float, sys: int
    ) -> Optional[Union[NavigationData, GlonassEphemeris, str]]:
        """
        读取RINEX导航数据体
        
        参考: readrnxnavb() in rinex.c
        关键逻辑：
        - GLONASS: i>=15 时 decode_geph，返回 GlonassEphemeris
        - SBAS/IRNSS: i>=15 时读完块，返回 None（位置钟差未实现）
        - GPS/Galileo/BDS/QZSS: i>=31 时完成，返回 NavigationData
        - EOF: 返回'EOF'字符串标记
        """
        data = []
        sat_id = ''
        toc = None
        i = 0  # 数据计数器（对应C代码的i）
        sp = 3  # 数据起始位置
        
        while True:
            line = fp.readline()
            if not line:
                return 'EOF'  # 返回特殊标记表示EOF
            
            if i == 0:
                # 第一行：卫星ID + TOC + 钟差参数
                if ver >= 3.0:
                    sat_id = line[0:3].strip()
                    sp = 4
                else:
                    prn = int(line[0:2].strip())
                    if sys == 0:  # GPS
                        sat_id = f"G{prn:02d}"
                    elif sys == 1:  # GLONASS
                        sat_id = f"R{prn:02d}"
                    elif sys == 2:  # Galileo
                        sat_id = f"E{prn:02d}"
                    elif sys == 4:  # BDS
                        sat_id = f"C{prn:02d}"
                    sp = 3
                
                # 解析TOC (Epoch)
                if ver >= 3.0:
                    # Format: " 2021 07 15 00 00 00"
                    toc = self._str_to_time_v3(line[sp:sp+19])
                else:
                    # Format: " 21  7 15  0  0  0.0"
                    toc = self._str_to_time_v2(line[sp:sp+19])
                
                if toc is None:
                    return None
                
                # 读取钟差参数 (3个)
                p = sp + 19
                for j in range(3):
                    val = self._fortran_float(line[p:p+19])
                    data.append(val)
                    i += 1
                    p += 19
                
            else:
                # 后续行：轨道参数
                p = sp
                for j in range(4):
                    if p + 19 <= len(line):
                        val = self._fortran_float(line[p:p+19])
                        data.append(val)
                        i += 1
                    else:
                        data.append(0.0)
                        i += 1
                    p += 19
                
                # 根据系统类型和参数数量判断是否读取完成
                sat_sys_char = sat_id[0] if len(sat_id) > 0 else 'G'
                
                # GLONASS: i>=15（对齐 decode_geph）
                if sat_sys_char == 'R' and i >= 15:
                    # GLONASS navigation epochs are UTC(SU); internal times are GPST.
                    # GPS-UTC is 18 s for this 2021 data set (and since 2017).
                    geph = self._decode_geph(ver, sat_id, float(toc) + 18.0, data)
                    return geph
                # SBAS/IRNSS: 已读完数据块，暂无对应星历类型
                if sat_sys_char in ['S', 'I'] and i >= 15:
                    return None
                
                # GPS/Galileo/BDS/QZSS: i>=31 时完成
                elif sat_sys_char in ['G', 'E', 'C', 'J'] and i >= 31:
                    # 解析卫星系统和PRN
                    prn = self._sat_id_to_prn(sat_id)
                    sat_sys = sat_id[0] if len(sat_id) > 0 else 'G'

                    # BDS 导航电文的历元时间（TOC）是 BDT，需要转换到 GPST（+14s）
                    # 对齐 compass-master/src/LibGnss/rinex.c: bdt2gpst(...)
                    toc_corrected = toc + 14.0 if sat_sys == 'C' else toc

                    # 创建NavigationData对象
                    nav = self._decode_eph(ver, sat_sys, prn, toc_corrected, data)
                    return nav
    
    def _decode_eph(self, ver: float, sat_sys: str, prn: int, 
                    toc: float, data: List[float]) -> NavigationData:
        """
        解码GPS/GAL/BDS星历
        
        参考: decode_eph() in rinex.c
        
        数据索引（data[]）:
        [0-2]   : af0, af1, af2 (钟差参数)
        [3]     : IODE
        [4]     : Crs
        [5]     : Delta n
        [6]     : M0
        [7]     : Cuc
        [8]     : e (偏心率)
        [9]     : Cus
        [10]    : sqrt(A)
        [11]    : Toe
        [12]    : Cic
        [13]    : OMEGA0
        [14]    : Cis
        [15]    : i0
        [16]    : Crc
        [17]    : omega
        [18]    : OMEGA DOT
        [19]    : IDOT
        [20]    : Codes on L2
        [21]    : GPS Week
        [22]    : L2 P data flag
        [23]    : SV accuracy
        [24]    : SV health
        [25]    : TGD
        [26]    : IODC
        [27]    : Transmission time
        [28]    : Fit interval (GPS) / AODC (BDS)
        [29-30] : spare
        """
        
        # 计算 TOE（从 GPS epoch 1980-01-06 起算的连续秒）
        week    = int(data[21])
        toe_sow = data[11]   # Toe，周内秒

        if sat_sys == 'C':
            # BDS 使用 BDT（北斗时）：
            #   BDT week 从 2006-01-01 起算，对应 GPS epoch 偏移 = 1356 GPS 周。
            #   GPST = BDT + 14 s（历史跳秒常数，自 BDT 建立以来恒定）。
            # 同时在 satellite.py 的 Ω 计算中会将 toe 减去 14 s 再取模，
            # 恢复出 BDT 周内秒（Ω₀ 的参考基准）。
            week    += 1356
            toe_sow += 14.0   # BDT → GPST

        toe_timestamp = week * 604800.0 + toe_sow
        
        # 创建NavigationData
        nav = NavigationData(
            sat_id=prn,
            system=self._sys_char_to_name(sat_sys),
            epoch=toc,
            toe=toe_timestamp,
            toc=toc,
            sqrt_a=data[10],
            e=data[8],
            i0=data[15],
            omega0=data[13],
            omega=data[17],
            M0=data[6],
            delta_n=data[5],
            i_dot=data[19],
            omega_dot=data[18],
            cuc=data[7],
            cus=data[9],
            crc=data[16],
            crs=data[4],
            cic=data[12],
            cis=data[14],
            af0=data[0],
            af1=data[1],
            af2=data[2],
            tgd=data[25],
            tgd2=(data[26] if sat_sys in ('E', 'C') else 0.0),
            health=int(data[24]),
            ura=data[23]
        )
        
        return nav

    def _decode_geph(
        self, ver: float, sat_id: str, toc: float, data: List[float]
    ) -> Optional[GlonassEphemeris]:
        """解码 GLONASS 广播星历（对齐 RTKLIB rinex.c decode_geph）。"""
        prn = self._sat_id_to_prn(sat_id)
        taun = -data[0]
        gamn = data[1]
        pos = (data[3] * 1e3, data[7] * 1e3, data[11] * 1e3)
        vel = (data[4] * 1e3, data[8] * 1e3, data[12] * 1e3)
        acc = (data[5] * 1e3, data[9] * 1e3, data[13] * 1e3)
        svh = int(data[6])
        frq = int(data[10])
        if frq > 128:
            frq -= 256
        age = int(data[14])
        MINFREQ_GLO, MAXFREQ_GLO = -7, 13
        if frq < MINFREQ_GLO or frq > MAXFREQ_GLO:
            return None
        return GlonassEphemeris(
            sat_id=prn,
            system="GLONASS",
            toe=toc,
            toc=toc,
            taun=taun,
            gamn=gamn,
            pos=pos,
            vel=vel,
            acc=acc,
            svh=svh,
            frq=frq,
            age=age,
        )
    
    def _str_to_time_v3(self, s: str) -> Optional[float]:
        """
        解析RINEX 3.x时间字符串
        Format: " 2021 07 15 00 00 00"
        """
        try:
            s = s.strip()
            parts = s.split()
            if len(parts) < 6:
                return None
            
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            second = float(parts[5])
            
            dt = datetime(year, month, day, hour, minute, int(second))
            # 加上小数秒
            dt += timedelta(seconds=(second - int(second)))
            
            # 转换为GPS时间（从1980-01-06 00:00:00起算的秒数）
            gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
            gps_seconds = (dt - gps_epoch).total_seconds()
            return gps_seconds
        except:
            return None
    
    def _str_to_time_v2(self, s: str) -> Optional[float]:
        """
        解析RINEX 2.x时间字符串
        Format: " 21  7 15  0  0  0.0"
        """
        try:
            year = int(s[0:3].strip())
            month = int(s[3:6].strip())
            day = int(s[6:9].strip())
            hour = int(s[9:12].strip())
            minute = int(s[12:15].strip())
            second = float(s[15:].strip())
            
            # 2位年份转换
            if year < 80:
                year += 2000
            else:
                year += 1900
            
            dt = datetime(year, month, day, hour, minute, int(second))
            dt += timedelta(seconds=(second - int(second)))
            
            # 转换为GPS时间（从1980-01-06 00:00:00起算的秒数）
            gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
            gps_seconds = (dt - gps_epoch).total_seconds()
            return gps_seconds
        except:
            return None
    
    def _fortran_float(self, s: str) -> float:
        """
        解析Fortran格式浮点数（支持D替代E）
        """
        s = s.strip()
        if not s or s == '':
            return 0.0
        
        # 替换D为E
        s = s.replace('D', 'E').replace('d', 'e')
        
        try:
            return float(s)
        except:
            return 0.0
    
    def _char_to_sys(self, c: str) -> int:
        """字符转系统编号"""
        sys_map = {
            'G': 0,  # GPS
            'R': 1,  # GLONASS
            'E': 2,  # Galileo
            'J': 3,  # QZSS
            'C': 4,  # BDS
            'I': 5,  # IRNSS
            'S': 6,  # SBAS
            'M': 7,  # Mixed
        }
        return sys_map.get(c, 0)
    
    def _sys_name(self, sys: int) -> str:
        """系统编号转名称"""
        names = ['GPS', 'GLONASS', 'Galileo', 'QZSS', 'BDS', 'IRNSS', 'SBAS', 'Mixed']
        return names[sys] if 0 <= sys < len(names) else 'Unknown'
    
    def _sys_char_to_name(self, c: str) -> str:
        """系统字符转名称"""
        name_map = {
            'G': 'GPS',
            'R': 'GLONASS',
            'E': 'Galileo',
            'J': 'QZSS',
            'C': 'BDS',
            'I': 'IRNSS',
            'S': 'SBAS'
        }
        return name_map.get(c, 'GPS')
    
    def _sat_id_to_prn(self, sat_id: str) -> int:
        """卫星ID转PRN"""
        if len(sat_id) < 2:
            return 0
        try:
            return int(sat_id[1:])
        except:
            return 0
    
    def read_obs(self, obs_file: str, max_epochs: int = 0, start_sow: float = 0.0) -> List[GNSSRawObservation]:
        """
        读取RINEX观测文件
        
        Args:
            obs_file: 观测文件路径
        
        Returns:
            观测数据列表（按历元）
        """
        if not os.path.exists(obs_file):
            raise FileNotFoundError(f"Obs file not found: {obs_file}")
        
        observations = []
        
        with open(obs_file, 'r') as fp:
            # 读取文件头
            header_info = self._read_obs_header(fp)
            self.obs_receiver_antenna = header_info["receiver_antenna"]
            self.obs_antenna_delta_enu = header_info["antenna_delta_enu"]
            
            # 读取观测数据体
            observations = self._read_obs_body(fp, header_info, max_epochs=max_epochs, start_sow=start_sow)
        
        self._normalize_doppler_convention(observations)
        return observations

    @staticmethod
    def _normalize_doppler_convention(observations: List[GNSSRawObservation]) -> bool:
        """Normalize receiver-specific Doppler signs to D = -dL/dt."""
        votes = []
        for current, following in zip(observations[:5], observations[1:6]):
            dt = following.timestamp - current.timestamp
            if dt <= 0.0 or dt > 300.0:
                continue
            next_sats = {(obs.system, obs.sat_id): obs for obs in following.observations}
            for obs in current.observations:
                nxt = next_sats.get((obs.system, obs.sat_id))
                if nxt is None or obs.cycle_slip or nxt.cycle_slip:
                    continue
                for phase_name, doppler_name in (
                    ("carrier_phase_L1", "doppler_L1"),
                    ("carrier_phase_L2", "doppler_L2"),
                ):
                    phase0, phase1 = getattr(obs, phase_name), getattr(nxt, phase_name)
                    doppler0, doppler1 = getattr(obs, doppler_name), getattr(nxt, doppler_name)
                    if any(value in (None, 0.0) for value in (phase0, phase1, doppler0, doppler1)):
                        continue
                    phase_rate = (float(phase1) - float(phase0)) / dt
                    mean_doppler = 0.5 * (float(doppler0) + float(doppler1))
                    if abs(mean_doppler) < 0.1:
                        continue
                    ratio = abs(phase_rate / mean_doppler)
                    if 0.5 <= ratio <= 1.5:
                        votes.append(np.sign(phase_rate * mean_doppler))

        reverse = len(votes) >= 4 and float(np.median(votes)) > 0.0
        if not reverse:
            return False

        for epoch in observations:
            for obs in epoch.observations:
                if obs.doppler_L1 is not None:
                    obs.doppler_L1 = -obs.doppler_L1
                if obs.doppler_L2 is not None:
                    obs.doppler_L2 = -obs.doppler_L2
                if obs.raw_observations:
                    for name, value in list(obs.raw_observations.items()):
                        if name.startswith("D") and value[0] is not None:
                            obs.raw_observations[name] = (-value[0], value[1], value[2])
        return True

    def _read_obs_header(self, fp) -> dict:
        """
        读取RINEX观测文件头
        
        Returns:
            dict包含：
            - version: RINEX版本
            - obs_types: 各系统观测类型 {system: [obs_types]}
            - approx_pos: 接收机概略位置 [x,y,z]
            - interval: 采样间隔
        """
        header = {
            'version': 3.0,
            'obs_types': {},  # {system_char: [C1C, L1C, ...]}
            'approx_pos': None,
            'interval': 1.0,
            'time_first': None,
            'time_last': None,
            'receiver_antenna': '',
            'antenna_delta_enu': None
        }
        
        while True:
            line = fp.readline()
            if not line:
                break
            
            if len(line) < 60:
                continue
            
            label = line[60:].strip()
            
            # RINEX版本
            if 'RINEX VERSION' in label:
                header['version'] = float(line[0:9].strip())
            
            # 系统观测类型 (RINEX 3.x)
            elif 'SYS / # / OBS TYPES' in label:
                sys_char = line[0]
                num_obs = int(line[3:6].strip())
                obs_list = []
                
                # 读取观测类型（每行最多13个类型）
                for i in range(0, num_obs, 13):
                    if i > 0:
                        line = fp.readline()
                    
                    start_col = 7 if i == 0 else 7
                    for j in range(13):
                        if len(obs_list) >= num_obs:
                            break
                        col = start_col + j * 4
                        if col + 3 <= len(line):
                            obs_type = line[col:col+3].strip()
                            if obs_type:
                                obs_list.append(obs_type)
                
                header['obs_types'][sys_char] = obs_list
            
            # 观测类型 (RINEX 2.x)
            elif '# / TYPES OF OBSERV' in label:
                num_obs = int(line[0:6].strip())
                obs_list = []
                
                # 读取观测类型
                for i in range(0, num_obs, 9):
                    if i > 0:
                        line = fp.readline()
                    
                    for j in range(9):
                        if len(obs_list) >= num_obs:
                            break
                        col = 10 + j * 6
                        if col + 2 <= len(line):
                            obs_type = line[col:col+2].strip()
                            if obs_type:
                                obs_list.append(obs_type)
                
                # RINEX 2.x所有系统共用观测类型
                for sys_char in ['G', 'R', 'E', 'C']:
                    header['obs_types'][sys_char] = obs_list
            
            # 接收机概略位置
            elif 'APPROX POSITION XYZ' in label:
                x = float(line[0:14].strip())
                y = float(line[14:28].strip())
                z = float(line[28:42].strip())
                header['approx_pos'] = np.array([x, y, z])

            elif 'ANT # / TYPE' in label:
                header['receiver_antenna'] = line[20:40].split()[0]

            elif 'ANTENNA: DELTA H/E/N' in label:
                h = float(line[0:14].strip())
                e = float(line[14:28].strip())
                n = float(line[28:42].strip())
                header['antenna_delta_enu'] = np.array([e, n, h])
            
            # 采样间隔
            elif 'INTERVAL' in label:
                header['interval'] = float(line[0:10].strip())
            
            # 文件头结束
            elif 'END OF HEADER' in label:
                break
        
        return header
    
    def _read_obs_body(self, fp, header: dict, max_epochs: int = 0, start_sow: float = 0.0) -> List[GNSSRawObservation]:
        """
        读取RINEX观测数据体
        
        Args:
            fp: 文件句柄
            header: 文件头信息
        """
        observations = []
        version = header['version']
        obs_types = header['obs_types']
        
        while True:
            line = fp.readline()
            if not line:
                break
            
            # 检查是否为历元标记行
            if version >= 3.0:
                if line[0] != '>':
                    continue
                    
                # 解析历元标记: > 2021 07 15 08 44 56.0000000  0 36
                epoch_time, flag, num_sat = self._parse_epoch_v3(line)
                if epoch_time is None:
                    continue

                # Event flags 2-5 use the count field for special records, not
                # satellites. Consume those records verbatim before resuming.
                if flag in (2, 3, 4, 5):
                    for _ in range(num_sat):
                        if not fp.readline():
                            break
                    continue
                
                # 读取该历元的所有卫星观测
                sat_obs_list = []
                week, sow = self._datetime_to_gps_time(epoch_time)
                if start_sow > 0.0 and sow < start_sow:
                    for _ in range(num_sat):
                        sat_line = fp.readline()
                        if not sat_line:
                            break
                        sys_obs_types = obs_types.get(sat_line[0], [])
                        self._read_sat_record_v3(fp, sat_line, len(sys_obs_types))
                    continue

                for i in range(num_sat):
                    sat_line = fp.readline()
                    if not sat_line:
                        break

                    sys_char = sat_line[0]
                    sys_obs_types = obs_types.get(sys_char, [])
                    merged = self._read_sat_record_v3(fp, sat_line, len(sys_obs_types))

                    # 解析卫星观测数据
                    sat_obs = self._parse_sat_obs_v3(merged, obs_types)
                    if sat_obs:
                        sat_obs_list.append(sat_obs)
                
                # 创建GNSS原始观测数据
                if sat_obs_list:
                    gnss_obs = GNSSRawObservation(
                        timestamp=sow,
                        week=week,
                        observations=sat_obs_list,
                        approx_position=header['approx_pos']
                    )
                    observations.append(gnss_obs)
                    if max_epochs > 0 and len(observations) >= max_epochs:
                        break
            
            else:
                # RINEX 2.x格式（暂不实现）
                pass
        
        return observations
    
    @staticmethod
    def _read_sat_record_v3(fp, first_line: str, num_obs_types: int) -> str:
        """Read one RINEX 3 satellite record without consuming the next record.

        Some producers emit standard continuation lines, while others emit one
        long line and trim trailing blank observation slots. Line length alone
        therefore cannot determine how many physical lines belong to a record.
        """
        merged = first_line[:3] + first_line[3:].rstrip("\r\n")
        required_length = 3 + 16 * num_obs_types

        while len(merged) < required_length:
            position = fp.tell()
            continuation = fp.readline()
            if not continuation:
                break

            if continuation.startswith("   "):
                merged += continuation[3:].rstrip("\r\n")
                continue

            fp.seek(position)
            break

        return merged

    def _parse_epoch_v3(self, line: str) -> Tuple[Optional[datetime], int, int]:
        """
        解析RINEX 3.x历元标记行
        
        格式: > 2021 07 15 08 44 56.0000000  0 36
        
        Returns:
            (epoch_time, flag, num_sat)
        """
        try:
            year = int(line[2:6].strip())
            month = int(line[7:9].strip())
            day = int(line[10:12].strip())
            hour = int(line[13:15].strip())
            minute = int(line[16:18].strip())
            second = float(line[19:29].strip())
            
            flag = int(line[31:32].strip())
            num_sat = int(line[33:36].strip())
            
            # 创建datetime对象
            sec_int = int(second)
            microsec = int((second - sec_int) * 1e6)
            epoch_time = datetime(year, month, day, hour, minute, sec_int, microsec)
            
            return epoch_time, flag, num_sat
        except Exception as e:
            print(f"Error parsing epoch: {e}")
            return None, 0, 0
    
    def _parse_sat_obs_v3(self, line: str, obs_types: dict) -> Optional[SatelliteObservation]:
        """
        解析RINEX 3.x卫星观测行
        
        格式: G18  24229632.117  24229641.316...
        
        Args:
            line: 卫星观测数据行
            obs_types: 观测类型字典 {system: [obs_types]}
        """
        if len(line) < 3:
            return None
        
        # 解析卫星ID
        sat_id_str = line[0:3].strip()
        if len(sat_id_str) < 2:
            return None
        
        system = sat_id_str[0]
        try:
            sat_num = int(sat_id_str[1:])
        except:
            return None
        
        # 获取该系统的观测类型
        sys_obs_types = obs_types.get(system, [])
        if not sys_obs_types:
            return None
        
        # 解析观测值（每个16字符）
        obs_values = {}
        for i, obs_type in enumerate(sys_obs_types):
            col_start = 3 + i * 16
            col_end = col_start + 14
            
            if col_end > len(line):
                break
            
            value_str = line[col_start:col_end].strip()
            lli = 0
            snr = 0.0
            if col_end < len(line):
                lli_str = line[col_end:col_end+1].strip()
                if lli_str:
                    try:
                        lli = int(lli_str)
                    except:
                        lli = 0
            if col_end + 1 < len(line):
                snr_str = line[col_end+1:col_end+2].strip()
                if snr_str:
                    try:
                        snr = float(snr_str) * 0.25
                    except:
                        snr = 0.0

            if value_str:
                try:
                    obs_values[obs_type] = (float(value_str), lli, snr)
                except:
                    obs_values[obs_type] = (0.0, lli, snr)

        obs_codes_c = {
            k: float(v[0])
            for k, v in obs_values.items()
            if k.startswith("C") and v[0] and float(v[0]) > 0.0 and np.isfinite(float(v[0]))
        }

        code_pr1: Optional[str] = None
        code_pr2: Optional[str] = None

        # 构造 SatelliteObservation 对象：对齐 RTKLIB 的“频点槽位”选码
        if system == "C":
            # BDS: slot0=B1(B1I/B1C)。第二频点默认优先 slot1(B2I/B2b,C7*)。
            # 若缺失，优先回退到 slot3(B3,C6*)（广播TGD常以B3为基准，更可控），再到 slot2(B2a,C5*), slot4(B2ab,C8*).
            pseudo_l1, lli_l1, snr_l1, code_pr1 = self._select_obs_with_code(
                obs_values, system, "C", 0
            )

            pseudo_l2 = 0.0
            lli_l2 = 0
            snr_l2 = 0.0
            code_pr2 = None
            for slot in (1, 3, 2, 4):
                v2, lli2, snr2, c2 = self._select_obs_with_code(
                    obs_values, system, "C", slot
                )
                if v2 and v2 != 0.0:
                    pseudo_l2, lli_l2, snr_l2, code_pr2 = v2, lli2, snr2, c2
                    break
        else:
            pseudo_l1, lli_l1, snr_l1 = self._select_obs(obs_values, system, "C", 0)
            pseudo_l2, lli_l2, snr_l2 = self._select_obs(obs_values, system, "C", 1)
        carrier_l1, lli_l1p, snr_l1p = self._select_obs(obs_values, system, 'L', 0)
        carrier_l2, lli_l2p, snr_l2p = self._select_obs(obs_values, system, 'L', 1)
        doppler_l1, _, _ = self._select_obs(obs_values, system, 'D', 0)
        doppler_l2, _, _ = self._select_obs(obs_values, system, 'D', 1)

        if snr_l1 == 0.0 and snr_l1p > 0.0:
            snr_l1 = snr_l1p
        if snr_l2 == 0.0 and snr_l2p > 0.0:
            snr_l2 = snr_l2p

        lli_l1 = lli_l1 if lli_l1 != 0 else lli_l1p
        lli_l2 = lli_l2 if lli_l2 != 0 else lli_l2p
        
        if pseudo_l1 == 0.0:
            return None
        
        sat_obs = SatelliteObservation(
            sat_id=sat_num,
            system=system,
            pseudorange_L1=pseudo_l1,
            pseudorange_L2=pseudo_l2 if pseudo_l2 != 0.0 else None,
            carrier_phase_L1=carrier_l1,
            carrier_phase_L2=carrier_l2,
            doppler_L1=doppler_l1,
            doppler_L2=doppler_l2,
            snr_L1=snr_l1,
            snr_L2=snr_l2,
            lli_L1=lli_l1,
            lli_L2=lli_l2,
            code_pr1=code_pr1,
            code_pr2=code_pr2,
            obs_codes_c=obs_codes_c if obs_codes_c else None,
            raw_observations=obs_values.copy(),
        )
        
        return sat_obs
    
    def _datetime_to_gps_time(self, dt: datetime) -> Tuple[int, float]:
        """
        将datetime转换为GPS时间(week, sow)
        
        Args:
            dt: datetime对象（GPS时间）
        
        Returns:
            (week, sow)
        """
        # GPS起始时间: 1980-01-06 00:00:00
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        
        # 计算总秒数
        delta = dt - gps_epoch
        total_seconds = delta.total_seconds()
        
        # 计算GPS周和周内秒
        week = int(total_seconds // 604800)
        sow = total_seconds % 604800
        
        return week, sow
