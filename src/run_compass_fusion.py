#!/usr/bin/env python3
"""CompassFusion XML entry point for GNSS/INS processing."""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from compass.ins import (
    read_gnss_pv_csv,
    read_imu_file,
    read_inertial_explorer_attitudes,
    read_inertial_explorer_pv,
    read_range_measurements_csv,
    initialize_state_from_gnss,
)
from compass.ins.coupling import INSFilterConfig, LooselyCoupledINS, TightlyCoupledINS
from compass.ins.mechanization import attitude_from_ie_hpr_rfu


@dataclass(frozen=True)
class RawGnssInputs:
    rinexo: tuple[Path, ...] = ()
    rinexn: Path | None = None
    sp3: Path | None = None
    clk: Path | None = None
    atx: Path | None = None
    eop: Path | None = None
    blq: Path | None = None
    sinex: Path | None = None
    bias: Path | None = None
    upd: Path | None = None


@dataclass(frozen=True)
class ProcessSettings:
    systems: tuple[str, ...] = ("GPS", "GAL", "BDS")
    excluded_sats: tuple[str, ...] = ()
    minimum_elev_deg: float = 7.0
    min_sat: int = 4
    max_res_norm: float = 4.0
    obs_weight: str = "SINEL"
    obs_combination: str = "RAW_MIX"
    frequency: int = 2
    phase: bool = True
    doppler: bool = False
    iono: bool = False
    tropo: bool = False
    pos_kin: bool = True
    basepos: str = "CFILE"
    slip_model: str = "default"


@dataclass(frozen=True)
class AmbiguitySettings:
    fix_mode: str = "SEARCH"
    part_fix: bool = True
    part_fix_num: int = 4
    ratio: float = 1.5
    min_common_time: int = 0
    baseline_length_limit_m: float = 3500.0
    widelane_interval_s: float = 30.0


@dataclass(frozen=True)
class AlignmentSettings:
    type: str = "POS"
    position_vector_m: float = 3.0
    velocity_vector_mps: float = 3.0
    coarse_align_time_s: float = 300.0


@dataclass(frozen=True)
class InitialStates:
    position_type: str = "OFF"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity_type: str = "OFF"
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attitude_type: str = "OFF"
    attitude_pitch_roll_yaw_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias_type: str = "OFF"
    gyro_bias_deg_h: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_bias_type: str = "OFF"
    accel_bias_mg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_scale_type: str = "OFF"
    gyro_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    accel_scale_type: str = "OFF"
    accel_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class OutputSettings:
    separator: str = ","
    precision_position: int = 4
    precision_velocity: int = 5
    include_velocity: bool = True
    include_covariance: bool = True
    include_bias: bool = True
    include_residual: bool = True


@dataclass(frozen=True)
class ConfigIssue:
    level: str
    message: str


@dataclass
class CompassFusionConfig:
    config_path: Path
    start_sow: float
    end_sow: float
    mode: str
    imu_path: Path
    imu_axis_order: str
    imu_gyro_unit: str
    imu_accel_unit: str
    imu_frequency_hz: float
    gnss_path: Path
    gnss_format: str
    range_path: Path | None
    range_format: str
    raw_gnss: RawGnssInputs
    attitude_path: Path | None
    attitude_mode: str
    truth_path: Path | None
    output_path: Path
    output_rate_hz: float
    outputs: OutputSettings
    fixed_only: bool
    process: ProcessSettings
    ambiguity: AmbiguitySettings
    alignment: AlignmentSettings
    initial_states: InitialStates
    position_sigma_m: float
    velocity_sigma_mps: float
    attitude_sigma_deg: float
    gyro_bias_sigma_rad_s: float
    accel_bias_sigma_mps2: float
    gyro_noise_rad_s: float
    accel_noise_mps2: float
    gyro_bias_rw_rad_s2: float
    accel_bias_rw_mps3: float
    min_position_sigma_m: float
    min_velocity_sigma_mps: float
    gate_sigma: float
    lever_arm_body_m: tuple[float, float, float]
    gnss_type: str
    gnss_frequency_hz: float
    gnss_delay_time_s: float
    gnss_min_sat: int
    gnss_max_pdop: float
    gnss_max_norm: float
    gnss_use_rtk_float: bool
    estimate_attitude_bias: bool
    velocity_attitude_aiding: bool
    velocity_attitude_min_speed_mps: float
    velocity_attitude_gain: float
    tight_range_gate_sigma: float
    tight_min_ranges: int
    tight_auto_init_clock: bool
    tight_position_ls_blend: float


def _text(node: ET.Element | None, default: str = "") -> str:
    if node is None or node.text is None:
        return default
    return node.text.strip().strip('"').strip("'")


def _attr(node: ET.Element | None, name: str, default: str = "") -> str:
    if node is None:
        return default
    return node.attrib.get(name, default).strip().strip('"').strip("'")


def _float_text(node: ET.Element | None, default: float = 0.0) -> float:
    value = _text(node)
    if not value:
        return default
    return float(value.split(",")[0].strip())


def _float_attr(node: ET.Element | None, name: str, default: float = 0.0) -> float:
    value = _attr(node, name)
    if not value:
        return default
    return float(value.split(",")[0].strip())


def _int_text(node: ET.Element | None, default: int = 0) -> int:
    return int(round(_float_text(node, float(default))))


def _int_attr(node: ET.Element | None, name: str, default: int = 0) -> int:
    return int(round(_float_attr(node, name, float(default))))


def _bool_text(node: ET.Element | None, default: bool = False) -> bool:
    value = _text(node)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on", "y"}


def _bool_attr(node: ET.Element | None, name: str, default: bool = False) -> bool:
    value = _attr(node, name)
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on", "y"}


def _words(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())


def _float_tuple(value: str, default: tuple[float, ...], count: int = 3) -> tuple[float, ...]:
    parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if not parts:
        return default
    try:
        values = tuple(float(part) for part in parts[:count])
    except ValueError:
        return default
    if len(values) == 1 and count > 1:
        values = values * count
    if len(values) != count:
        return default
    return values


def _xyz_tuple(node: ET.Element | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if node is None:
        return default
    if {"X", "Y", "Z"} & set(node.attrib):
        return (
            _float_attr(node, "X", default[0]),
            _float_attr(node, "Y", default[1]),
            _float_attr(node, "Z", default[2]),
        )
    values = _float_tuple(_text(node), default, 3)
    return values[0], values[1], values[2]


def _node_value_float(node: ET.Element | None, default: float = 0.0) -> float:
    if node is None:
        return default
    return _float_attr(node, "Value", _float_text(node, default))


def _node_initial_std(node: ET.Element | None, default: float = 0.0) -> float:
    if node is None:
        return default
    return _float_tuple(_attr(node, "InitialSTD"), (default,), 1)[0]


def _deg_per_hour_to_rad_per_s(value: float) -> float:
    return math.radians(value) / 3600.0


def _mg_to_mps2(value: float) -> float:
    return value * 9.80665e-3


def _path(base: Path, value: str) -> Path:
    value = value.replace("\\", "/").strip()
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = (base / path).resolve()
    if candidate.exists():
        return candidate
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if path.parts and path.parts[0].lower() in {"data", "conference", "configs", "scripts", "compass"}:
        return cwd_candidate
    return candidate


def _optional_path(base: Path, node: ET.Element | None) -> Path | None:
    if node is None or not _text(node):
        return None
    return _path(base, _text(node))


def _path_list(base: Path, node: ET.Element | None) -> tuple[Path, ...]:
    if node is None:
        return ()
    return tuple(_path(base, value) for value in _text(node).splitlines() if value.strip())


def _lever_from_node(node: ET.Element | None) -> tuple[float, float, float]:
    if node is None:
        return (0.0, 0.0, 0.0)
    if node.attrib:
        return (
            _float_attr(node, "X", 0.0),
            _float_attr(node, "Y", 0.0),
            _float_attr(node, "Z", 0.0),
        )
    parts = [part.strip() for part in _text(node).split(",")]
    if len(parts) != 3:
        return (0.0, 0.0, 0.0)
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_raw_gnss_inputs(base: Path, inputs: ET.Element | None) -> RawGnssInputs:
    if inputs is None:
        return RawGnssInputs()
    return RawGnssInputs(
        rinexo=_path_list(base, inputs.find("rinexo")),
        rinexn=_optional_path(base, inputs.find("rinexn")),
        sp3=_optional_path(base, inputs.find("sp3")),
        clk=_optional_path(base, inputs.find("clk")),
        atx=_optional_path(base, inputs.find("atx")),
        eop=_optional_path(base, inputs.find("EOP")) or _optional_path(base, inputs.find("eop")),
        blq=_optional_path(base, inputs.find("blq")),
        sinex=_optional_path(base, inputs.find("sinex")),
        bias=_optional_path(base, inputs.find("bias")),
        upd=_optional_path(base, inputs.find("upd")),
    )


def _parse_process(root: ET.Element, gen: ET.Element | None) -> ProcessSettings:
    process = root.find("process")
    systems = _words(_text(process.find("sys") if process is not None else None)) if process is not None else ()
    if not systems:
        systems = _words(_text(gen.find("sys") if gen is not None else None)) if gen is not None else ()
    excluded = _words(_text(process.find("sat_rm") if process is not None else None)) if process is not None else ()
    if not excluded:
        excluded = _words(_text(gen.find("sat_rm") if gen is not None else None)) if gen is not None else ()
    return ProcessSettings(
        systems=systems or ("GPS", "GAL", "BDS"),
        excluded_sats=excluded,
        minimum_elev_deg=_float_text(process.find("minimum_elev") if process is not None else None, 7.0),
        min_sat=_int_text(process.find("min_sat") if process is not None else None, 4),
        max_res_norm=_float_text(process.find("max_res_norm") if process is not None else None, 4.0),
        obs_weight=_text(process.find("obs_weight") if process is not None else None, "SINEL"),
        obs_combination=_text(process.find("obs_combination") if process is not None else None, "RAW_MIX"),
        frequency=_int_text(process.find("frequency") if process is not None else None, 2),
        phase=_bool_text(process.find("phase") if process is not None else None, True),
        doppler=_bool_text(process.find("doppler") if process is not None else None, False),
        iono=_bool_text(process.find("iono") if process is not None else None, False),
        tropo=_bool_text(process.find("tropo") if process is not None else None, False),
        pos_kin=_bool_text(process.find("pos_kin") if process is not None else None, True),
        basepos=_text(process.find("basepos") if process is not None else None, "CFILE"),
        slip_model=_text(process.find("slip_model") if process is not None else None, "default"),
    )


def _parse_ambiguity(root: ET.Element) -> AmbiguitySettings:
    ambiguity = root.find("ambiguity")
    return AmbiguitySettings(
        fix_mode=_text(ambiguity.find("fix_mode") if ambiguity is not None else None, "SEARCH"),
        part_fix=_bool_text(ambiguity.find("part_fix") if ambiguity is not None else None, True),
        part_fix_num=_int_text(ambiguity.find("part_fix_num") if ambiguity is not None else None, 4),
        ratio=_float_text(ambiguity.find("ratio") if ambiguity is not None else None, 1.5),
        min_common_time=_int_text(ambiguity.find("min_common_time") if ambiguity is not None else None, 0),
        baseline_length_limit_m=_float_text(ambiguity.find("baseline_length_limit") if ambiguity is not None else None, 3500.0),
        widelane_interval_s=_float_text(ambiguity.find("widelane_interval") if ambiguity is not None else None, 30.0),
    )


def _parse_alignment(ins: ET.Element | None) -> AlignmentSettings:
    alignment = ins.find("Alignment") if ins is not None else None
    return AlignmentSettings(
        type=_attr(alignment, "Type", "POS"),
        position_vector_m=_node_value_float(alignment.find("PositionVector") if alignment is not None else None, 3.0),
        velocity_vector_mps=_node_value_float(alignment.find("VelocityVector") if alignment is not None else None, 3.0),
        coarse_align_time_s=_node_value_float(alignment.find("CoarseAlignTime") if alignment is not None else None, 300.0),
    )


def _parse_initial_states(ins: ET.Element | None) -> InitialStates:
    initial = ins.find("InitialStates") if ins is not None else None
    position = initial.find("Position") if initial is not None else None
    velocity = initial.find("Velocity") if initial is not None else None
    attitude = initial.find("Attitude") if initial is not None else None
    gyro_bias = initial.find("GyroBias") if initial is not None else None
    accel_bias = initial.find("AcceBias") if initial is not None else None
    gyro_scale = initial.find("GyroScale") if initial is not None else None
    accel_scale = initial.find("AcceScale") if initial is not None else None
    return InitialStates(
        position_type=_attr(position, "Type", "OFF"),
        position=_xyz_tuple(position, (0.0, 0.0, 0.0)),
        velocity_type=_attr(velocity, "Type", "OFF"),
        velocity=_xyz_tuple(velocity, (0.0, 0.0, 0.0)),
        attitude_type=_attr(attitude, "Type", "OFF"),
        attitude_pitch_roll_yaw_deg=(
            _float_attr(attitude, "Pitch", 0.0),
            _float_attr(attitude, "Roll", 0.0),
            _float_attr(attitude, "Yaw", 0.0),
        ),
        gyro_bias_type=_attr(gyro_bias, "Type", "OFF"),
        gyro_bias_deg_h=_xyz_tuple(gyro_bias, (0.0, 0.0, 0.0)),
        accel_bias_type=_attr(accel_bias, "Type", "OFF"),
        accel_bias_mg=_xyz_tuple(accel_bias, (0.0, 0.0, 0.0)),
        gyro_scale_type=_attr(gyro_scale, "Type", "OFF"),
        gyro_scale=_xyz_tuple(gyro_scale, (1.0, 1.0, 1.0)),
        accel_scale_type=_attr(accel_scale, "Type", "OFF"),
        accel_scale=_xyz_tuple(accel_scale, (1.0, 1.0, 1.0)),
    )


def _parse_outputs(outputs: ET.Element | None) -> OutputSettings:
    options = outputs.find("options") if outputs is not None else None
    return OutputSettings(
        separator=_attr(options, "Separator", ",") or ",",
        precision_position=_int_attr(options, "PositionPrecision", 4),
        precision_velocity=_int_attr(options, "VelocityPrecision", 5),
        include_velocity=_bool_attr(options, "Velocity", True),
        include_covariance=_bool_attr(options, "Covariance", True),
        include_bias=_bool_attr(options, "Bias", True),
        include_residual=_bool_attr(options, "Residual", True),
    )


def _nearest_attitude(attitudes, time: float, max_dt: float = 1.0):
    if not attitudes:
        return None
    nearest = min(attitudes, key=lambda item: abs(item.time - time))
    if abs(nearest.time - time) > max_dt:
        return None
    return nearest


def _enabled(value: str) -> bool:
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _delimiter(value: str) -> str:
    if value.lower() in {"tab", "\\t"}:
        return "\t"
    if value.lower() in {"space", "blank"}:
        return " "
    return value[0] if value else ","


def _existing_optional_paths(config: CompassFusionConfig) -> tuple[tuple[str, Path | None], ...]:
    return (
        ("inputs.ranges", config.range_path),
        ("inputs.attitude", config.attitude_path),
        ("inputs.truth", config.truth_path),
        ("inputs.rinexn", config.raw_gnss.rinexn),
        ("inputs.sp3", config.raw_gnss.sp3),
        ("inputs.clk", config.raw_gnss.clk),
        ("inputs.atx", config.raw_gnss.atx),
        ("inputs.eop", config.raw_gnss.eop),
        ("inputs.blq", config.raw_gnss.blq),
        ("inputs.sinex", config.raw_gnss.sinex),
        ("inputs.bias", config.raw_gnss.bias),
        ("inputs.upd", config.raw_gnss.upd),
    )


def validate_config(config: CompassFusionConfig, check_files: bool = True) -> list[ConfigIssue]:
    issues: list[ConfigIssue] = []
    supported_modes = {"loose", "tight", "mechanization"}
    supported_gnss_formats = {"csv", "ie"}
    supported_range_formats = {"csv"}
    supported_attitude_modes = {"init", "epoch"}

    if config.mode not in supported_modes:
        issues.append(ConfigIssue("error", f"unsupported <gen><mode>: {config.mode}; supported: loose, mechanization"))
    if config.gnss_format not in supported_gnss_formats:
        issues.append(ConfigIssue("error", f"unsupported <inputs><gnss format>: {config.gnss_format}; supported: csv, ie"))
    if config.range_format not in supported_range_formats:
        issues.append(ConfigIssue("error", f"unsupported <inputs><ranges format>: {config.range_format}; supported: csv"))
    if config.mode == "tight" and config.range_path is None:
        issues.append(ConfigIssue("error", "mode='tight' requires <inputs><ranges format='csv'>"))
    if config.attitude_mode not in supported_attitude_modes:
        issues.append(ConfigIssue("error", f"unsupported <inputs><attitude mode>: {config.attitude_mode}; supported: init, epoch"))
    if config.attitude_mode == "epoch" and config.attitude_path is None:
        issues.append(ConfigIssue("error", "attitude mode='epoch' requires <inputs><attitude>"))
    if config.end_sow and config.start_sow and config.end_sow <= config.start_sow:
        issues.append(ConfigIssue("error", "end_sow must be greater than start_sow"))
    if config.output_rate_hz < 0.0:
        issues.append(ConfigIssue("error", "outputs/rate_hz must be >= 0"))
    if config.outputs.precision_position < 0 or config.outputs.precision_velocity < 0:
        issues.append(ConfigIssue("error", "output precision values must be >= 0"))
    if not 0.0 <= config.velocity_attitude_gain <= 1.0:
        issues.append(ConfigIssue("error", "VelocityAttitudeAiding Gain must be between 0 and 1"))
    if config.gnss_min_sat < 0:
        issues.append(ConfigIssue("error", "LCISetting MinSat must be >= 0"))
    if config.process.minimum_elev_deg < -90.0 or config.process.minimum_elev_deg > 90.0:
        issues.append(ConfigIssue("error", "process/minimum_elev must be within [-90, 90] deg"))

    if config.gnss_type.upper() == "TCI" and config.range_path is None:
        issues.append(ConfigIssue("warning", "TCI is configured, but no <inputs><ranges> file is set; raw RINEX TCI front-end is not wired yet"))
    if config.raw_gnss.rinexo or config.raw_gnss.rinexn:
        if config.mode == "loose":
            message = "raw RINEX/product inputs are parsed for future PPP/RTK/TCI, but current loose mode uses <inputs><gnss> PV only"
        else:
            message = "raw RINEX/product inputs are parsed for future PPP/RTK/TCI, but this entry currently needs prebuilt PV/range files"
        issues.append(ConfigIssue("warning", message))
    if config.ambiguity.fix_mode.upper() != "OFF":
        issues.append(ConfigIssue("warning", "ambiguity settings are parsed but not used until the raw RTK/PPP front-end is connected"))
    if _enabled(config.initial_states.gyro_scale_type) or _enabled(config.initial_states.accel_scale_type):
        issues.append(ConfigIssue("warning", "IMU scale-factor initial states are parsed but not applied by the current mechanization"))

    if check_files:
        required = (("inputs.imu", config.imu_path), ("inputs.gnss", config.gnss_path))
        for label, path in required:
            if not path.exists():
                issues.append(ConfigIssue("error", f"{label} does not exist: {path}"))
        for label, path in _existing_optional_paths(config):
            if path is not None and not path.exists():
                issues.append(ConfigIssue("warning", f"{label} does not exist: {path}"))
        for index, path in enumerate(config.raw_gnss.rinexo, start=1):
            if not path.exists():
                issues.append(ConfigIssue("warning", f"inputs.rinexo[{index}] does not exist: {path}"))

    return issues


def _format_issues(issues: list[ConfigIssue]) -> str:
    if not issues:
        return "Config check: OK"
    return "\n".join(f"{issue.level.upper()}: {issue.message}" for issue in issues)


def _has_errors(issues: list[ConfigIssue]) -> bool:
    return any(issue.level == "error" for issue in issues)


def config_summary(config: CompassFusionConfig) -> str:
    raw_count = len(config.raw_gnss.rinexo) + sum(
        path is not None
        for path in (
            config.raw_gnss.rinexn,
            config.raw_gnss.sp3,
            config.raw_gnss.clk,
            config.raw_gnss.atx,
            config.raw_gnss.eop,
            config.raw_gnss.blq,
            config.raw_gnss.sinex,
            config.raw_gnss.bias,
            config.raw_gnss.upd,
        )
    )
    active = [
        "CompassFusion configuration",
        f"  config        : {config.config_path}",
        f"  mode          : {config.mode} / GNSS integration={config.gnss_type}",
        f"  time          : {config.start_sow:g} -> {config.end_sow:g} sow",
        f"  imu           : {config.imu_path}",
        f"  imu format    : axis={config.imu_axis_order}, gyro={config.imu_gyro_unit}, accel={config.imu_accel_unit}, freq={config.imu_frequency_hz:g} Hz",
        f"  gnss pv       : {config.gnss_path} ({config.gnss_format}, fixed_only={config.fixed_only})",
        f"  ranges        : {config.range_path or 'none'} ({config.range_format})",
        f"  attitude      : {config.attitude_path or 'none'} ({config.attitude_mode})",
        f"  output        : {config.output_path} @ {config.output_rate_hz:g} Hz",
        f"  lever arm RFU : {config.lever_arm_body_m[0]:.4f}, {config.lever_arm_body_m[1]:.4f}, {config.lever_arm_body_m[2]:.4f} m",
        f"  init sigma    : pos={config.position_sigma_m:g} m, vel={config.velocity_sigma_mps:g} m/s, att={config.attitude_sigma_deg:g} deg",
        f"  noise         : gyro={config.gyro_noise_rad_s:g} rad/s, accel={config.accel_noise_mps2:g} m/s^2",
        f"  process       : systems={' '.join(config.process.systems)}, elev>={config.process.minimum_elev_deg:g} deg, min_sat={config.process.min_sat}",
        f"  raw inputs    : {raw_count} configured (reserved for PPP/RTK/TCI front-end)",
    ]
    return "\n".join(active)


def apply_cli_overrides(config: CompassFusionConfig, args: argparse.Namespace) -> CompassFusionConfig:
    if getattr(args, "mode", None):
        config.mode = args.mode.lower()
    if getattr(args, "out", None):
        config.output_path = Path(args.out).resolve()
    if getattr(args, "start_sow", None) is not None:
        config.start_sow = float(args.start_sow)
    if getattr(args, "end_sow", None) is not None:
        config.end_sow = float(args.end_sow)
    if getattr(args, "output_rate", None) is not None:
        config.output_rate_hz = float(args.output_rate)
    if getattr(args, "attitude_mode", None):
        config.attitude_mode = args.attitude_mode.lower()
    if getattr(args, "fixed_only", None) is not None:
        config.fixed_only = bool(args.fixed_only)
    if getattr(args, "velocity_attitude_aiding", None) is not None:
        config.velocity_attitude_aiding = bool(args.velocity_attitude_aiding)
    if getattr(args, "velocity_attitude_min_speed", None) is not None:
        config.velocity_attitude_min_speed_mps = float(args.velocity_attitude_min_speed)
    if getattr(args, "velocity_attitude_gain", None) is not None:
        config.velocity_attitude_gain = float(args.velocity_attitude_gain)
    return config


def load_config(path: str | Path) -> CompassFusionConfig:
    config_path = Path(path).resolve()
    root = ET.parse(config_path).getroot()
    base = config_path.parent

    gen = root.find("gen")
    inputs = root.find("inputs")
    outputs = root.find("outputs")
    ins = root.find("ins")
    integration = root.find("integration")
    gnss_integration = integration.find("GNSS") if integration is not None else None
    estimator = integration.find("Estimator") if integration is not None else None
    filter_node = root.find("filter")

    data_format = ins.find("DataFormat") if ins is not None else None
    axis_node = data_format.find("AxisOrder") if data_format is not None else None
    gyro_node = data_format.find("GyroUnit") if data_format is not None else None
    accel_node = data_format.find("AcceUnit") if data_format is not None else None
    imu_freq_node = data_format.find("Frequency") if data_format is not None else None

    imu_node = inputs.find("imu") if inputs is not None else None
    gnss_node = inputs.find("gnss") if inputs is not None else None
    range_node = inputs.find("ranges") if inputs is not None else None
    attitude_node = inputs.find("attitude") if inputs is not None else None
    truth_node = inputs.find("truth") if inputs is not None else None
    output_node = outputs.find("ins") if outputs is not None else None

    if imu_node is None or gnss_node is None:
        raise ValueError("config requires <inputs><imu> and <inputs><gnss>")

    lever_node = gnss_integration.find("AntennaLever") if gnss_integration is not None else None
    vel_aid_node = gnss_integration.find("VelocityAttitudeAiding") if gnss_integration is not None else None
    lci_node = gnss_integration.find("LCISetting") if gnss_integration is not None else None
    gnss_freq_node = gnss_integration.find("Frequency") if gnss_integration is not None else None
    gnss_delay_node = gnss_integration.find("DelayTime") if gnss_integration is not None else None

    process = _parse_process(root, gen)
    ambiguity = _parse_ambiguity(root)
    alignment = _parse_alignment(ins)
    initial_states = _parse_initial_states(ins)
    outputs_config = _parse_outputs(outputs)

    return CompassFusionConfig(
        config_path=config_path,
        start_sow=_float_text(gen.find("start_sow") if gen is not None else None, 0.0),
        end_sow=_float_text(gen.find("end_sow") if gen is not None else None, 0.0),
        mode=_text(gen.find("mode") if gen is not None else None, "loose").lower(),
        imu_path=_path(base, _text(imu_node)),
        imu_axis_order=_attr(axis_node, "Type", "rfu").lower(),
        imu_gyro_unit=_attr(gyro_node, "Type", "rad/s").lower(),
        imu_accel_unit=_attr(accel_node, "Type", "mps2").lower(),
        imu_frequency_hz=_node_value_float(imu_freq_node, 0.0),
        gnss_path=_path(base, _text(gnss_node)),
        gnss_format=_attr(gnss_node, "format", "csv").lower(),
        range_path=_path(base, _text(range_node)) if range_node is not None and _text(range_node) else None,
        range_format=_attr(range_node, "format", "csv").lower() if range_node is not None else "csv",
        raw_gnss=_parse_raw_gnss_inputs(base, inputs),
        attitude_path=_path(base, _text(attitude_node)) if attitude_node is not None and _text(attitude_node) else None,
        attitude_mode=_attr(attitude_node, "mode", "init").lower() if attitude_node is not None else "init",
        truth_path=_path(base, _text(truth_node)) if truth_node is not None and _text(truth_node) else None,
        output_path=_path(base, _text(output_node, "data/compass_fusion_result.csv")),
        output_rate_hz=_float_text(outputs.find("rate_hz") if outputs is not None else None, 1.0),
        outputs=outputs_config,
        fixed_only=_bool_text(gen.find("fixed_only") if gen is not None else None, False),
        process=process,
        ambiguity=ambiguity,
        alignment=alignment,
        initial_states=initial_states,
        position_sigma_m=_float_attr(estimator.find("Position") if estimator is not None else None, "InitialSTD", 5.0),
        velocity_sigma_mps=_float_attr(estimator.find("Velocity") if estimator is not None else None, "InitialSTD", 2.0),
        attitude_sigma_deg=_float_attr(estimator.find("Attitude") if estimator is not None else None, "InitialSTD", 20.0),
        gyro_bias_sigma_rad_s=_float_attr(filter_node, "initial_gyro_bias_sigma_rad_s", _deg_per_hour_to_rad_per_s(
            _node_initial_std(estimator.find("GyroBias") if estimator is not None else None, 1.0)
        )),
        accel_bias_sigma_mps2=_float_attr(filter_node, "initial_accel_bias_sigma_mps2", _mg_to_mps2(
            _node_initial_std(estimator.find("AcceBias") if estimator is not None else None, 1.0)
        )),
        gyro_noise_rad_s=_float_attr(filter_node, "gyro_noise_rad_s", 5.0e-4),
        accel_noise_mps2=_float_attr(filter_node, "accel_noise_mps2", 2.0e-2),
        gyro_bias_rw_rad_s2=_float_attr(filter_node, "gyro_bias_rw_rad_s2", 1.0e-6),
        accel_bias_rw_mps3=_float_attr(filter_node, "accel_bias_rw_mps3", 1.0e-4),
        min_position_sigma_m=_float_attr(filter_node, "min_position_sigma_m", 0.05),
        min_velocity_sigma_mps=_float_attr(filter_node, "min_velocity_sigma_mps", 0.08),
        gate_sigma=_float_attr(estimator, "GateSigma", 1000.0),
        lever_arm_body_m=_lever_from_node(lever_node),
        gnss_type=_attr(gnss_integration, "Type", "LCI"),
        gnss_frequency_hz=_node_value_float(gnss_freq_node, 1.0),
        gnss_delay_time_s=_node_value_float(gnss_delay_node, 0.0),
        gnss_min_sat=_int_attr(lci_node, "MinSat", process.min_sat),
        gnss_max_pdop=_float_attr(lci_node, "MaxPDOP", 6.0),
        gnss_max_norm=_float_attr(lci_node, "MaxNorm", process.max_res_norm),
        gnss_use_rtk_float=_bool_attr(lci_node, "UseRTKFloatSolution", True),
        estimate_attitude_bias=_bool_attr(estimator, "EstimateAttitudeBias", False),
        velocity_attitude_aiding=_bool_attr(vel_aid_node, "Enabled", False),
        velocity_attitude_min_speed_mps=_float_attr(vel_aid_node, "MinSpeed", 2.0),
        velocity_attitude_gain=_float_attr(vel_aid_node, "Gain", 1.0),
        tight_range_gate_sigma=_float_attr(estimator, "TightRangeGateSigma", 15.0),
        tight_min_ranges=_int_attr(estimator, "TightMinRanges", 4),
        tight_auto_init_clock=_bool_attr(estimator, "TightAutoInitClock", True),
        tight_position_ls_blend=_float_attr(estimator, "TightPositionLSBlend", 0.0),
    )


def _read_gnss(config: CompassFusionConfig):
    if config.gnss_format == "ie":
        return read_inertial_explorer_pv(config.gnss_path, fixed_only=config.fixed_only)
    if config.gnss_format == "csv":
        return read_gnss_pv_csv(config.gnss_path, fixed_only=config.fixed_only)
    raise ValueError(f"unsupported gnss format: {config.gnss_format}")


def _read_ranges(config: CompassFusionConfig) -> dict[float, list]:
    if config.range_path is None:
        return {}
    if config.range_format == "csv":
        return read_range_measurements_csv(config.range_path)
    raise ValueError(f"unsupported ranges format: {config.range_format}")


def _summarize(result_path: Path, truth_path: Path, delimiter: str = ",") -> str:
    truth = {round(item.time): item.position for item in read_inertial_explorer_pv(truth_path)}
    errors = []
    with result_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter=delimiter):
            try:
                sow = round(float(row["sow"]))
                position = np.asarray([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
            except (KeyError, ValueError):
                continue
            if sow in truth:
                errors.append(float(np.linalg.norm(position - truth[sow])))
    if not errors:
        return "matched=0"
    arr = np.asarray(errors, dtype=float)
    return (
        f"matched={len(arr)} "
        f"3D median/RMS/P95/MAX="
        f"{np.median(arr):.4f}/"
        f"{math.sqrt(float(np.mean(arr * arr))):.4f}/"
        f"{np.percentile(arr, 95):.4f}/"
        f"{np.max(arr):.4f} m"
    )


def run_config(config: CompassFusionConfig, verbose: bool = False) -> int:
    issues = validate_config(config)
    if verbose:
        print(config_summary(config))
        if issues:
            print(_format_issues(issues), file=sys.stderr)
    if _has_errors(issues):
        raise SystemExit(_format_issues(issues))

    gnss = _read_gnss(config)
    if config.start_sow:
        gnss = [item for item in gnss if item.time >= config.start_sow]
    if config.end_sow:
        gnss = [item for item in gnss if item.time <= config.end_sow]
    if not gnss:
        raise SystemExit("No usable GNSS measurements")
    ranges_by_time = _read_ranges(config)
    if config.mode == "tight":
        if not ranges_by_time:
            raise SystemExit("No usable tight range measurements")
        ranges_by_time = {
            time: measurements
            for time, measurements in ranges_by_time.items()
            if (not config.start_sow or time >= config.start_sow) and (not config.end_sow or time <= config.end_sow)
        }
        if not ranges_by_time:
            raise SystemExit("No tight range measurements in requested time span")

    start = config.start_sow or gnss[0].time
    end = config.end_sow or gnss[-1].time
    imu = read_imu_file(
        config.imu_path,
        start_sow=start,
        end_sow=end,
        gyro_unit=config.imu_gyro_unit,
        axis_order=config.imu_axis_order,
    )
    if not imu:
        raise SystemExit("No usable IMU samples")

    first = gnss[0]
    init_window = [sample for sample in imu if first.time <= sample.time <= first.time + 1.0]
    initial = initialize_state_from_gnss(first.time, first.position, first.velocity, init_window)
    if _enabled(config.initial_states.position_type):
        initial.position = np.asarray(config.initial_states.position, dtype=float)
    if _enabled(config.initial_states.velocity_type):
        initial.velocity = np.asarray(config.initial_states.velocity, dtype=float)
    if _enabled(config.initial_states.attitude_type):
        pitch, roll, yaw = config.initial_states.attitude_pitch_roll_yaw_deg
        initial.C_be = attitude_from_ie_hpr_rfu(initial.position, yaw, pitch, roll)
    if _enabled(config.initial_states.gyro_bias_type):
        initial.gyro_bias = np.asarray([_deg_per_hour_to_rad_per_s(v) for v in config.initial_states.gyro_bias_deg_h], dtype=float)
    if _enabled(config.initial_states.accel_bias_type):
        initial.accel_bias = np.asarray([_mg_to_mps2(v) for v in config.initial_states.accel_bias_mg], dtype=float)
    attitudes = read_inertial_explorer_attitudes(config.attitude_path) if config.attitude_path else []
    if config.attitude_path:
        attitude = _nearest_attitude(attitudes, first.time)
        if attitude is None:
            raise SystemExit("No IE attitude found near initial GNSS epoch")
        initial.C_be = attitude_from_ie_hpr_rfu(initial.position, attitude.heading_deg, attitude.pitch_deg, attitude.roll_deg)

    filter_cls = TightlyCoupledINS if config.mode == "tight" else LooselyCoupledINS
    filt = filter_cls(
        initial,
        INSFilterConfig(
            gyro_noise_rad_s=config.gyro_noise_rad_s,
            accel_noise_mps2=config.accel_noise_mps2,
            gyro_bias_rw_rad_s2=config.gyro_bias_rw_rad_s2,
            accel_bias_rw_mps3=config.accel_bias_rw_mps3,
            initial_position_sigma_m=config.position_sigma_m,
            initial_velocity_sigma_mps=config.velocity_sigma_mps,
            initial_attitude_sigma_rad=np.deg2rad(config.attitude_sigma_deg),
            initial_gyro_bias_sigma_rad_s=config.gyro_bias_sigma_rad_s,
            initial_accel_bias_sigma_mps2=config.accel_bias_sigma_mps2,
            min_position_sigma_m=config.min_position_sigma_m,
            min_velocity_sigma_mps=config.min_velocity_sigma_mps,
            measurement_gate_sigma=config.gate_sigma,
            lever_arm_body_m=config.lever_arm_body_m,
            estimate_attitude_bias=config.estimate_attitude_bias,
            velocity_attitude_aiding=config.velocity_attitude_aiding,
            velocity_attitude_min_speed_mps=config.velocity_attitude_min_speed_mps,
            velocity_attitude_gain=config.velocity_attitude_gain,
            tight_range_gate_sigma=config.tight_range_gate_sigma,
            tight_min_ranges=config.tight_min_ranges,
            tight_auto_init_clock=config.tight_auto_init_clock,
            tight_position_ls_blend=config.tight_position_ls_blend,
        ),
    )

    output_period = 0.0 if config.output_rate_hz <= 0.0 else 1.0 / config.output_rate_hz
    next_output = start
    gnss_index = 0
    range_epochs = sorted(ranges_by_time)
    range_index = 0
    written = 0
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    delimiter = _delimiter(config.outputs.separator)
    header = ["sow", "x_m", "y_m", "z_m"]
    if config.outputs.include_velocity:
        header.extend(["vx_mps", "vy_mps", "vz_mps"])
    if config.outputs.include_covariance:
        header.extend(["sdx_m", "sdy_m", "sdz_m", "sdv_mps"])
    header.extend(["mode", "gnss_used"])
    if config.outputs.include_residual:
        header.append("innovation_norm")
    if config.outputs.include_bias:
        header.extend(["gyro_bias_x", "gyro_bias_y", "gyro_bias_z", "accel_bias_x", "accel_bias_y", "accel_bias_z"])
    with config.output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter=delimiter)
        writer.writerow(header)
        for sample in imu:
            if sample.time < filt.state.time:
                continue
            snapshot = filt.propagate(sample)
            if config.mode == "loose":
                while gnss_index < len(gnss) and gnss[gnss_index].time <= sample.time + 0.5 * max(sample.dt, 0.0):
                    measurement = gnss[gnss_index]
                    if measurement.time >= filt.state.time - max(sample.dt, 0.0):
                        if config.attitude_mode == "epoch":
                            attitude = _nearest_attitude(attitudes, measurement.time)
                            if attitude is not None:
                                filt.state.C_be = attitude_from_ie_hpr_rfu(
                                    filt.state.position,
                                    attitude.heading_deg,
                                    attitude.pitch_deg,
                                    attitude.roll_deg,
                                )
                        snapshot = filt.update_pv(measurement)
                    gnss_index += 1
            elif config.mode == "tight":
                while range_index < len(range_epochs) and range_epochs[range_index] <= sample.time + 0.5 * max(sample.dt, 0.0):
                    epoch_time = range_epochs[range_index]
                    if epoch_time >= filt.state.time - max(sample.dt, 0.0):
                        snapshot = filt.update_ranges(ranges_by_time[epoch_time])
                    range_index += 1
            if output_period == 0.0 or sample.time + 1.0e-9 >= next_output:
                diag = np.sqrt(np.maximum(np.diag(snapshot.covariance), 0.0))
                row = [
                    f"{snapshot.time:.3f}",
                    *(f"{value:.{config.outputs.precision_position}f}" for value in snapshot.position),
                ]
                if config.outputs.include_velocity:
                    row.extend(f"{value:.{config.outputs.precision_velocity}f}" for value in snapshot.velocity)
                if config.outputs.include_covariance:
                    row.extend(f"{value:.4f}" for value in diag[:3])
                    row.append(f"{float(np.mean(diag[3:6])):.4f}")
                row.extend([snapshot.mode, snapshot.gnss_used])
                if config.outputs.include_residual:
                    row.append(f"{snapshot.innovation_norm:.3f}")
                if config.outputs.include_bias:
                    row.extend(f"{value:.8g}" for value in filt.state.gyro_bias)
                    row.extend(f"{value:.8g}" for value in filt.state.accel_bias)
                writer.writerow(row)
                written += 1
                next_output += output_period if output_period > 0.0 else 0.0

    print(
        f"Finished CompassFusion {config.mode}: imu_samples={len(imu)} "
        f"gnss={len(gnss)} range_epochs={len(ranges_by_time)} rows={written} output={config.output_path}"
    )
    if config.truth_path:
        print(_summarize(config.output_path, config.truth_path, delimiter))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="CompassFusion XML config")
    parser.add_argument("--mode", choices=["loose", "tight", "mechanization"], help="override <gen><mode>")
    parser.add_argument("--out", help="override <outputs><ins>")
    parser.add_argument("--start-sow", type=float, help="override <gen><start_sow>")
    parser.add_argument("--end-sow", type=float, help="override <gen><end_sow>")
    parser.add_argument("--output-rate", type=float, help="override <outputs><rate_hz>")
    parser.add_argument("--attitude-mode", choices=["init", "epoch"], help="override <inputs><attitude mode>")
    parser.add_argument("--fixed-only", dest="fixed_only", action="store_const", const=True, default=None, help="use fixed GNSS/PV epochs only")
    parser.add_argument("--no-fixed-only", dest="fixed_only", action="store_const", const=False, help="use all valid GNSS/PV epochs")
    parser.add_argument(
        "--velocity-attitude-aiding",
        dest="velocity_attitude_aiding",
        action="store_const",
        const=True,
        default=None,
        help="enable velocity-based attitude aiding",
    )
    parser.add_argument(
        "--no-velocity-attitude-aiding",
        dest="velocity_attitude_aiding",
        action="store_const",
        const=False,
        help="disable velocity-based attitude aiding",
    )
    parser.add_argument("--velocity-attitude-min-speed", type=float, help="override VelocityAttitudeAiding MinSpeed")
    parser.add_argument("--velocity-attitude-gain", type=float, help="override VelocityAttitudeAiding Gain")
    parser.add_argument("--check", action="store_true", help="validate config and exit without processing")
    parser.add_argument("--strict", action="store_true", help="treat config warnings as failures when used with --check")
    parser.add_argument("--verbose", action="store_true", help="print applied config summary")
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    if args.check:
        issues = validate_config(config)
        if args.verbose:
            print(config_summary(config))
        print(_format_issues(issues))
        failed = _has_errors(issues) or (args.strict and bool(issues))
        return 1 if failed else 0
    return run_config(config, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
