#!/usr/bin/env python3
"""Run CompassFusion loose/tight regression tests and write plots."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.ins.coupling import read_inertial_explorer_pv
from run_compass_fusion import load_config, run_config


def _read_result(path: Path, updated_only: bool = False) -> dict[float, np.ndarray]:
    rows: dict[float, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                if updated_only and int(float(row.get("gnss_used", "0"))) <= 0:
                    continue
                rows[float(row["sow"])] = np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
            except (KeyError, ValueError):
                continue
    return rows


def _last_sow(path: Path) -> float:
    last = 0.0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                last = float(row["sow"])
            except (KeyError, ValueError):
                continue
    return last


def _read_ie_truth(path: Path) -> dict[float, np.ndarray]:
    return {round(item.time, 3): item.position for item in read_inertial_explorer_pv(path)}


def _match_errors(result: dict[float, np.ndarray], truth: dict[float, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    times = []
    errors = []
    truth_by_int = {round(time): position for time, position in truth.items()}
    for time, position in result.items():
        key = round(time)
        if key in truth_by_int:
            times.append(float(time))
            errors.append(float(np.linalg.norm(position - truth_by_int[key])))
    return np.asarray(times), np.asarray(errors)


def _stats(errors: np.ndarray) -> dict[str, float]:
    if errors.size == 0:
        return {"matched": 0}
    return {
        "matched": float(errors.size),
        "median_m": float(np.median(errors)),
        "rms_m": float(math.sqrt(np.mean(errors * errors))),
        "p95_m": float(np.percentile(errors, 95.0)),
        "p99_m": float(np.percentile(errors, 99.0)),
        "max_m": float(np.max(errors)),
    }


def _write_stats(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "mode", "matched", "median_m", "rms_m", "p95_m", "p99_m", "max_m", "output"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _plot_errors(out_dir: Path, case: str, times: np.ndarray, errors: np.ndarray) -> None:
    if errors.size == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_time = times - times[0] if times.size else times

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rel_time, errors, lw=1.2)
    ax.set_title(f"{case} 3D position error")
    ax.set_xlabel("Time from start (s)")
    ax.set_ylabel("3D error (m)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{case}_error_timeseries.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, sorted_errors.size + 1) / sorted_errors.size
    ax.plot(sorted_errors, cdf, lw=1.5)
    ax.set_title(f"{case} error CDF")
    ax.set_xlabel("3D error (m)")
    ax.set_ylabel("CDF")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{case}_error_cdf.png", dpi=160)
    plt.close(fig)


def _set_xml_text(root: ET.Element, path: str, value: str) -> None:
    node = root.find(path)
    if node is None:
        raise ValueError(f"missing XML node: {path}")
    node.text = value


def _write_loose_config(source: Path, target: Path, output: Path, start_sow: float, end_sow: float) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    _set_xml_text(root, "gen/mode", "loose")
    _set_xml_text(root, "gen/start_sow", f"{start_sow:.3f}")
    _set_xml_text(root, "gen/end_sow", f"{end_sow:.3f}")
    _set_xml_text(root, "outputs/ins", str(output.resolve()))
    target.parent.mkdir(parents=True, exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=True)


def _write_tight_synthetic_case(case_dir: Path, duration_s: int = 30) -> tuple[Path, Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    imu = case_dir / "imu.txt"
    gnss = case_dir / "gnss.csv"
    ranges = case_dir / "ranges.csv"
    config = case_dir / "tight_synthetic.xml"
    output = case_dir / "tight_synthetic.csv"
    truth = case_dir / "truth.csv"
    true_position = np.array([6378137.0, 0.0, 0.0])
    sats = [
        np.array([15600000.0, 21700000.0, 20100000.0]),
        np.array([18700000.0, -13400000.0, 23200000.0]),
        np.array([-17600000.0, 14100000.0, 21700000.0]),
        np.array([21100000.0, 7000000.0, -14100000.0]),
        np.array([-15000000.0, -21000000.0, 12000000.0]),
    ]

    imu_lines = [f"{time} 0 0 0 9.7803253359 0 0" for time in range(duration_s + 1)]
    imu.write_text("\n".join(imu_lines) + "\n", encoding="utf-8")
    gnss.write_text(
        "sow,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,sdx_m,sdy_m,sdz_m,status\n"
        f"0,{true_position[0]},{true_position[1]},{true_position[2]},0,0,0,0.1,0.1,0.1,1\n",
        encoding="utf-8",
    )
    range_lines = ["sow,sat_x_m,sat_y_m,sat_z_m,pseudorange_m,variance_m2"]
    for time in range(duration_s + 1):
        for sat in sats:
            pr = float(np.linalg.norm(true_position - sat))
            range_lines.append(f"{time},{sat[0]},{sat[1]},{sat[2]},{pr},1")
    ranges.write_text("\n".join(range_lines) + "\n", encoding="utf-8")
    truth.write_text(
        "sow,x_m,y_m,z_m\n"
        + "\n".join(f"{time},{true_position[0]},{true_position[1]},{true_position[2]}" for time in range(duration_s + 1))
        + "\n",
        encoding="utf-8",
    )
    config.write_text(
        f"""<?xml version="1.0"?>
<config>
  <gen>
    <mode>tight</mode>
    <start_sow>0</start_sow>
    <end_sow>{duration_s}</end_sow>
  </gen>
  <inputs>
    <imu>{imu.name}</imu>
    <gnss format="csv">{gnss.name}</gnss>
    <ranges format="csv">{ranges.name}</ranges>
  </inputs>
  <outputs>
    <ins>{output.name}</ins>
    <rate_hz>1</rate_hz>
  </outputs>
  <integration>
    <Estimator GateSigma="1000">
      <Position InitialSTD="3"/>
      <Velocity InitialSTD="1"/>
      <Attitude InitialSTD="1"/>
    </Estimator>
  </integration>
</config>
""",
        encoding="utf-8",
    )
    return config, truth


def _write_spp_ins_config(
    case_dir: Path,
    mode: str,
    pv_path: Path,
    ranges_path: Path | None,
    output_path: Path,
    start_sow: float,
    end_sow: float,
) -> Path:
    config = case_dir / f"spp_ins_{mode}.xml"
    ranges_line = f'    <ranges format="csv">{ranges_path.resolve()}</ranges>\n' if ranges_path is not None else ""
    config.write_text(
        f"""<?xml version="1.0"?>
<config>
  <gen>
    <mode>{mode}</mode>
    <start_sow>{start_sow:.3f}</start_sow>
    <end_sow>{end_sow:.3f}</end_sow>
  </gen>
  <inputs>
    <imu>{(ROOT / "data/great_msf/MSF_20211013/IMU/smallimu_out_2.txt").resolve()}</imu>
    <gnss format="csv">{pv_path.resolve()}</gnss>
{ranges_line}    <attitude format="ie" mode="init">{(ROOT / "data/great_msf/MSF_20211013/groundtruth/groundtruth_211013_ADIS.txt").resolve()}</attitude>
    <truth format="ie">{(ROOT / "data/great_msf/MSF_20211013/groundtruth/groundtruth_211013_ADIS.txt").resolve()}</truth>
  </inputs>
  <outputs>
    <ins>{output_path.resolve()}</ins>
    <rate_hz>1</rate_hz>
  </outputs>
  <ins>
    <DataFormat>
      <AxisOrder Type="garfu"/>
      <GyroUnit Type="DPS"/>
      <AcceUnit Type="MPS2"/>
      <Frequency Value="100"/>
    </DataFormat>
  </ins>
  <integration>
    <GNSS Type="{"TCI" if mode == "tight" else "LCI"}">
      <AntennaLever Type="RFU" X="0.01" Y="-0.273" Z="0.09"/>
    </GNSS>
    <Estimator GateSigma="1000" EstimateAttitudeBias="false" TightPositionLSBlend="{"1.0" if mode == "tight" else "0.0"}">
      <Attitude InitialSTD="0.5"/>
      <Velocity InitialSTD="5"/>
      <Position InitialSTD="10"/>
    </Estimator>
  </integration>
</config>
""",
        encoding="utf-8",
    )
    return config


def _write_pv_ins_config(
    case_dir: Path,
    case_name: str,
    pv_path: Path,
    output_path: Path,
    start_sow: float,
    end_sow: float,
    attitude_mode: str = "init",
) -> Path:
    config = case_dir / f"{case_name}.xml"
    config.write_text(
        f"""<?xml version="1.0"?>
<config>
  <gen>
    <mode>loose</mode>
    <start_sow>{start_sow:.3f}</start_sow>
    <end_sow>{end_sow:.3f}</end_sow>
  </gen>
  <inputs>
    <imu>{(ROOT / "data/great_msf/MSF_20211013/IMU/smallimu_out_2.txt").resolve()}</imu>
    <gnss format="csv">{pv_path.resolve()}</gnss>
    <attitude format="ie" mode="{attitude_mode}">{(ROOT / "data/great_msf/MSF_20211013/groundtruth/groundtruth_211013_ADIS.txt").resolve()}</attitude>
    <truth format="ie">{(ROOT / "data/great_msf/MSF_20211013/groundtruth/groundtruth_211013_ADIS.txt").resolve()}</truth>
  </inputs>
  <outputs>
    <ins>{output_path.resolve()}</ins>
    <rate_hz>1</rate_hz>
  </outputs>
  <ins>
    <DataFormat>
      <AxisOrder Type="garfu"/>
      <GyroUnit Type="DPS"/>
      <AcceUnit Type="MPS2"/>
      <Frequency Value="100"/>
    </DataFormat>
  </ins>
  <integration>
    <GNSS Type="LCI">
      <AntennaLever Type="RFU" X="0.01" Y="-0.273" Z="0.09"/>
    </GNSS>
    <Estimator GateSigma="1000" EstimateAttitudeBias="false">
      <Attitude InitialSTD="0.5"/>
      <Velocity InitialSTD="1"/>
      <Position InitialSTD="3"/>
    </Estimator>
  </integration>
</config>
""",
        encoding="utf-8",
    )
    return config


def _prepare_real_spp_inputs(case_dir: Path, start_sow: float, end_sow: float, systems: str) -> tuple[Path, Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    pv = case_dir / f"spp_{systems.lower()}_pv.csv"
    ranges = case_dir / f"spp_{systems.lower()}_ranges.csv"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "export_spp_ins_inputs.py"),
        "--obs",
        str(ROOT / "data" / "great_msf" / "MSF_20211013" / "GNSS" / "SEPT2860.21O"),
        "--nav",
        str(ROOT / "data" / "great_msf" / "MSF_20211013" / "GNSS" / "brdm2860.21p"),
        "--pv-out",
        str(pv),
        "--ranges-out",
        str(ranges),
        "--start-sow",
        f"{start_sow:.3f}",
        "--end-sow",
        f"{end_sow:.3f}",
        "--elev",
        "10",
        "--systems",
        systems,
        "--max-epochs",
        str(max(1, int(round(end_sow - start_sow)) + 1)),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return pv, ranges


def _prepare_real_ppk_inputs(case_dir: Path, start_sow: float, max_epochs: int) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    out = case_dir / "ppk_gec.csv"
    cmd = [
        sys.executable,
        str(ROOT / "run_ppk.py"),
        "--rover",
        str(ROOT / "data" / "great_msf" / "MSF_20211013" / "GNSS" / "SEPT2860.21O"),
        "--base",
        str(ROOT / "data" / "great_msf" / "MSF_20211013" / "GNSS" / "R2932860.21o"),
        "--nav",
        str(ROOT / "data" / "great_msf" / "MSF_20211013" / "GNSS" / "brdm2860.21p"),
        "--sp3",
        str(ROOT / "data" / "great_msf" / "MSF_20211013" / "GNSS" / "COD0MGXFIN_20212860000_01D_05M_ORB.SP3"),
        "--clk",
        str(ROOT / "data" / "great_msf" / "MSF_20211013" / "GNSS" / "COD0MGXFIN_20212860000_01D_30S_CLK.CLK"),
        "--base-position=-2267776.6563,5009356.8058,3220981.5585",
        "--out",
        str(out),
        "--start-sow",
        f"{start_sow:.3f}",
        "--max-epochs",
        str(max_epochs),
        "--elev",
        "10",
        "--filter-mode",
        "multipass",
        "--progress-every",
        "0",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out


def _read_csv_truth(path: Path) -> dict[float, np.ndarray]:
    truth = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            truth[float(row["sow"])] = np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
    return truth


def _run_case(case: str, mode: str, config_path: Path, truth: dict[float, np.ndarray], plot_dir: Path) -> dict[str, float | str]:
    config = load_config(config_path)
    run_config(config)
    result = _read_result(config.output_path)
    times, errors = _match_errors(result, truth)
    _plot_errors(plot_dir, case, times, errors)
    row: dict[str, float | str] = {"case": case, "mode": mode, "output": str(config.output_path)}
    row.update(_stats(errors))
    return row


def _run_case_with_updated(case: str, mode: str, config_path: Path, truth: dict[float, np.ndarray], plot_dir: Path) -> list[dict[str, float | str]]:
    config = load_config(config_path)
    run_config(config)
    rows = []
    for suffix, updated_only in (("", False), ("_updated", True)):
        result = _read_result(config.output_path, updated_only=updated_only)
        times, errors = _match_errors(result, truth)
        plot_case = f"{case}{suffix}"
        _plot_errors(plot_dir, plot_case, times, errors)
        row: dict[str, float | str] = {"case": plot_case, "mode": mode, "output": str(config.output_path)}
        row.update(_stats(errors))
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/compass_fusion_tests"))
    parser.add_argument("--full", action="store_true", help="run full GREAT loose segment instead of a quick smoke segment")
    parser.add_argument("--real-spp", action="store_true", help="run real RINEX SPP-INS loose/tight tests")
    parser.add_argument("--real-spp-systems", default="GEC")
    parser.add_argument("--real-ppp", action="store_true", help="run existing SEPT2860 PPP-INS loose test")
    parser.add_argument("--real-ppk", action="store_true", help="run GREAT SEPT/R293 PPK-INS loose quick test")
    parser.add_argument("--skip-loose", action="store_true")
    parser.add_argument("--skip-tight", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    plot_dir = out_dir / "plots"
    rows: list[dict[str, float | str]] = []

    if not args.skip_loose:
        start, end = (289371.0, 293100.0) if args.full else (289371.0, 289430.0)
        loose_config = out_dir / "loose_great.xml"
        loose_output = out_dir / "loose_great.csv"
        _write_loose_config(ROOT / "configs" / "compass_fusion_211013.xml", loose_config, loose_output, start, end)
        truth = _read_ie_truth(ROOT / "data" / "great_msf" / "MSF_20211013" / "groundtruth" / "groundtruth_211013_ADIS.txt")
        rows.append(_run_case("loose_great", "loose", loose_config, truth, plot_dir))

    if not args.skip_tight:
        tight_config, tight_truth_path = _write_tight_synthetic_case(out_dir / "tight_synthetic")
        rows.append(_run_case("tight_synthetic", "tight", tight_config, _read_csv_truth(tight_truth_path), plot_dir))

    if args.real_spp:
        start, end = (289371.0, 293100.0) if args.full else (289371.0, 289430.0)
        real_dir = out_dir / f"real_spp_{args.real_spp_systems.lower()}"
        pv, ranges = _prepare_real_spp_inputs(real_dir, start, end, args.real_spp_systems)
        actual_end = _last_sow(pv) or end
        truth = _read_ie_truth(ROOT / "data" / "great_msf" / "MSF_20211013" / "groundtruth" / "groundtruth_211013_ADIS.txt")
        loose_config = _write_spp_ins_config(real_dir, "loose", pv, None, real_dir / "spp_ins_loose.csv", start, actual_end)
        rows.extend(_run_case_with_updated(f"real_spp_{args.real_spp_systems.lower()}_loose", "loose", loose_config, truth, plot_dir))
        tight_config = _write_spp_ins_config(real_dir, "tight", pv, ranges, real_dir / "spp_ins_tight.csv", start, actual_end)
        rows.extend(_run_case_with_updated(f"real_spp_{args.real_spp_systems.lower()}_tight", "tight", tight_config, truth, plot_dir))

    if args.real_ppp:
        start, end = (289371.0, 293100.0) if args.full else (289371.0, 289430.0)
        ppp_path = ROOT / "data" / "ppp_result_SEPT2860_multipass.csv"
        ppp_dir = out_dir / "real_ppp_gec"
        ppp_dir.mkdir(parents=True, exist_ok=True)
        truth = _read_ie_truth(ROOT / "data" / "great_msf" / "MSF_20211013" / "groundtruth" / "groundtruth_211013_ADIS.txt")
        ppp_config = _write_pv_ins_config(ppp_dir, "ppp_ins_loose", ppp_path, ppp_dir / "ppp_ins_loose.csv", start, end)
        rows.extend(_run_case_with_updated("real_ppp_gec_loose", "loose", ppp_config, truth, plot_dir))

    if args.real_ppk:
        start, end = (289371.0, 289970.0) if args.full else (289371.0, 289430.0)
        max_epochs = 600 if args.full else 60
        ppk_dir = out_dir / "real_ppk_gec"
        ppk_path = _prepare_real_ppk_inputs(ppk_dir, start, max_epochs)
        truth = _read_ie_truth(ROOT / "data" / "great_msf" / "MSF_20211013" / "groundtruth" / "groundtruth_211013_ADIS.txt")
        ppk_config = _write_pv_ins_config(ppk_dir, "ppk_ins_loose", ppk_path, ppk_dir / "ppk_ins_loose.csv", start, end)
        rows.extend(_run_case_with_updated("real_ppk_gec_loose", "loose", ppk_config, truth, plot_dir))

    _write_stats(out_dir / "summary.csv", rows)
    print(f"Wrote summary: {out_dir / 'summary.csv'}")
    print(f"Wrote plots: {plot_dir}")
    for row in rows:
        print(
            f"{row['case']} {row['mode']}: matched={int(float(row.get('matched', 0)))} "
            f"RMS={float(row.get('rms_m', float('nan'))):.4f} m "
            f"P95={float(row.get('p95_m', float('nan'))):.4f} m"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
