from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from run_compass_fusion import apply_cli_overrides, load_config, run_config, validate_config


def test_compass_fusion_sample_config_parses_great_style_blocks():
    config = load_config(Path("configs") / "compass_fusion_211013.xml")

    assert config.mode == "loose"
    assert config.imu_axis_order == "garfu"
    assert config.imu_gyro_unit == "dps"
    assert config.imu_frequency_hz == 100.0
    assert config.lever_arm_body_m == pytest.approx((0.01, -0.273, 0.09))
    assert config.process.systems == ("GPS", "GAL", "BDS")
    assert config.process.excluded_sats == ("C01", "C02", "C03", "C04", "C05")
    assert config.process.minimum_elev_deg == 7.0
    assert config.process.frequency == 2
    assert config.ambiguity.fix_mode == "SEARCH"
    assert config.ambiguity.part_fix is True
    assert config.alignment.type == "POS"
    assert config.initial_states.position_type == "OFF"
    assert config.outputs.separator == ","
    assert config.outputs.include_bias is True
    assert config.gyro_noise_rad_s == pytest.approx(5.0e-4)
    assert config.accel_noise_mps2 == pytest.approx(2.0e-2)
    assert len(config.raw_gnss.rinexo) == 2
    assert config.raw_gnss.rinexn is not None


def test_compass_fusion_config_parses_manual_initial_states(tmp_path):
    config_path = tmp_path / "fusion.xml"
    config_path.write_text(
        """<?xml version="1.0"?>
<config>
  <gen>
    <mode>loose</mode>
    <start_sow>1</start_sow>
    <end_sow>2</end_sow>
    <fixed_only>true</fixed_only>
  </gen>
  <inputs>
    <imu>imu.txt</imu>
    <gnss format="csv">gnss.csv</gnss>
  </inputs>
  <outputs>
    <ins>out.csv</ins>
    <rate_hz>10</rate_hz>
    <options Separator="tab" Velocity="false" Covariance="false" Residual="false" Bias="false"/>
  </outputs>
  <filter gyro_noise_rad_s="0.1" accel_noise_mps2="0.2"
          gyro_bias_rw_rad_s2="0.3" accel_bias_rw_mps3="0.4"
          min_position_sigma_m="0.5" min_velocity_sigma_mps="0.6"
          initial_gyro_bias_sigma_rad_s="0.7" initial_accel_bias_sigma_mps2="0.8"/>
  <ins>
    <DataFormat>
      <AxisOrder Type="rfu"/>
      <GyroUnit Type="rad/s"/>
      <AcceUnit Type="mps2"/>
      <Frequency Value="200"/>
    </DataFormat>
    <Alignment Type="VEL">
      <PositionVector Value="4"/>
      <VelocityVector Value="5"/>
      <CoarseAlignTime Value="6"/>
    </Alignment>
    <InitialStates>
      <Position Type="Cartesian" X="1" Y="2" Z="3"/>
      <Velocity Type="Cartesian" X="4" Y="5" Z="6"/>
      <Attitude Type="ON" Pitch="7" Roll="8" Yaw="9"/>
      <GyroBias Type="ON" X="10" Y="11" Z="12"/>
      <AcceBias Type="ON" X="13" Y="14" Z="15"/>
      <GyroScale Type="Const" X="1.1" Y="1.2" Z="1.3"/>
      <AcceScale Type="Const" X="1.4" Y="1.5" Z="1.6"/>
    </InitialStates>
  </ins>
  <integration>
    <GNSS Type="TCI">
      <AntennaLever Type="RFU" X="0.1" Y="0.2" Z="0.3"/>
      <Frequency Value="5"/>
      <DelayTime Value="0.01"/>
      <LCISetting MinSat="7" MaxPDOP="8" MaxNorm="9" UseRTKFloatSolution="OFF"/>
      <VelocityAttitudeAiding Enabled="true" MinSpeed="3" Gain="0.5"/>
    </GNSS>
    <Estimator Type="Forward" GateSigma="11" EstimateAttitudeBias="true">
      <Attitude InitialSTD="1"/>
      <Velocity InitialSTD="2"/>
      <Position InitialSTD="3"/>
      <GyroBias InitialSTD="4"/>
      <AcceBias InitialSTD="5"/>
    </Estimator>
  </integration>
</config>
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.fixed_only is True
    assert config.output_rate_hz == 10.0
    assert config.outputs.separator == "tab"
    assert config.outputs.include_velocity is False
    assert config.initial_states.position == pytest.approx((1.0, 2.0, 3.0))
    assert config.initial_states.velocity == pytest.approx((4.0, 5.0, 6.0))
    assert config.initial_states.attitude_pitch_roll_yaw_deg == pytest.approx((7.0, 8.0, 9.0))
    assert config.initial_states.gyro_bias_deg_h == pytest.approx((10.0, 11.0, 12.0))
    assert config.initial_states.accel_bias_mg == pytest.approx((13.0, 14.0, 15.0))
    assert config.initial_states.gyro_scale == pytest.approx((1.1, 1.2, 1.3))
    assert config.initial_states.accel_scale == pytest.approx((1.4, 1.5, 1.6))
    assert config.gyro_noise_rad_s == pytest.approx(0.1)
    assert config.accel_noise_mps2 == pytest.approx(0.2)
    assert config.gyro_bias_rw_rad_s2 == pytest.approx(0.3)
    assert config.accel_bias_rw_mps3 == pytest.approx(0.4)
    assert config.gnss_type == "TCI"
    assert config.gnss_frequency_hz == 5.0
    assert config.gnss_delay_time_s == 0.01
    assert config.gnss_min_sat == 7
    assert config.gnss_use_rtk_float is False
    assert config.estimate_attitude_bias is True
    assert config.velocity_attitude_aiding is True


def test_compass_fusion_validation_reports_bad_time_window(tmp_path):
    (tmp_path / "imu.txt").write_text("1 0 0 0 0 0 0\n", encoding="utf-8")
    (tmp_path / "gnss.csv").write_text("sow,x_m,y_m,z_m\n1,1,2,3\n", encoding="utf-8")
    config_path = tmp_path / "bad_time.xml"
    config_path.write_text(
        """<?xml version="1.0"?>
<config>
  <gen>
    <mode>loose</mode>
    <start_sow>20</start_sow>
    <end_sow>10</end_sow>
  </gen>
  <inputs>
    <imu>imu.txt</imu>
    <gnss format="csv">gnss.csv</gnss>
  </inputs>
</config>
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    issues = validate_config(config)

    assert config.imu_path == tmp_path / "imu.txt"
    assert config.gnss_path == tmp_path / "gnss.csv"
    assert any(issue.level == "error" and "end_sow" in issue.message for issue in issues)


def test_compass_fusion_validation_marks_tci_as_reserved(tmp_path):
    (tmp_path / "imu.txt").write_text("1 0 0 0 0 0 0\n", encoding="utf-8")
    (tmp_path / "gnss.csv").write_text("sow,x_m,y_m,z_m\n1,1,2,3\n", encoding="utf-8")
    config_path = tmp_path / "tci.xml"
    config_path.write_text(
        """<?xml version="1.0"?>
<config>
  <inputs>
    <imu>imu.txt</imu>
    <gnss format="csv">gnss.csv</gnss>
  </inputs>
  <integration>
    <GNSS Type="TCI"/>
  </integration>
</config>
""",
        encoding="utf-8",
    )

    issues = validate_config(load_config(config_path))

    assert any(issue.level == "warning" and "TCI" in issue.message for issue in issues)


def test_compass_fusion_cli_overrides_sample_config(tmp_path):
    config = load_config(Path("configs") / "compass_fusion_211013.xml")
    output = tmp_path / "override.csv"
    args = SimpleNamespace(
        out=str(output),
        start_sow=10.0,
        end_sow=20.0,
        output_rate=5.0,
        attitude_mode="epoch",
        fixed_only=True,
        velocity_attitude_aiding=True,
        velocity_attitude_min_speed=6.0,
        velocity_attitude_gain=0.25,
    )

    updated = apply_cli_overrides(config, args)

    assert updated.output_path == output.resolve()
    assert updated.start_sow == 10.0
    assert updated.end_sow == 20.0
    assert updated.output_rate_hz == 5.0
    assert updated.attitude_mode == "epoch"
    assert updated.fixed_only is True
    assert updated.velocity_attitude_aiding is True
    assert updated.velocity_attitude_min_speed_mps == 6.0
    assert updated.velocity_attitude_gain == 0.25


def test_compass_fusion_tight_mode_runs_with_range_csv(tmp_path):
    imu = tmp_path / "imu.txt"
    gnss = tmp_path / "gnss.csv"
    ranges = tmp_path / "ranges.csv"
    out = tmp_path / "tight_out.csv"
    config_path = tmp_path / "tight.xml"
    true_position = np.array([6378137.0, 0.0, 0.0])
    sats = [
        np.array([15600000.0, 21700000.0, 20100000.0]),
        np.array([18700000.0, -13400000.0, 23200000.0]),
        np.array([-17600000.0, 14100000.0, 21700000.0]),
        np.array([21100000.0, 7000000.0, -14100000.0]),
        np.array([-15000000.0, -21000000.0, 12000000.0]),
    ]

    imu.write_text("0 0 0 0 9.7803253359 0 0\n1 0 0 0 9.7803253359 0 0\n", encoding="utf-8")
    gnss.write_text(
        "sow,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,sdx_m,sdy_m,sdz_m\n"
        "0,6378137,0,0,0,0,0,0.1,0.1,0.1\n",
        encoding="utf-8",
    )
    range_lines = ["sow,sat_x_m,sat_y_m,sat_z_m,pseudorange_m,variance_m2"]
    for sat in sats:
        pr = float(np.linalg.norm(true_position - sat))
        range_lines.append(f"0,{sat[0]},{sat[1]},{sat[2]},{pr},1")
    ranges.write_text("\n".join(range_lines) + "\n", encoding="utf-8")
    config_path.write_text(
        f"""<?xml version="1.0"?>
<config>
  <gen>
    <mode>tight</mode>
    <start_sow>0</start_sow>
    <end_sow>1</end_sow>
  </gen>
  <inputs>
    <imu>{imu.name}</imu>
    <gnss format="csv">{gnss.name}</gnss>
    <ranges format="csv">{ranges.name}</ranges>
  </inputs>
  <outputs>
    <ins>{out.name}</ins>
    <rate_hz>1</rate_hz>
  </outputs>
  <integration>
    <Estimator GateSigma="1000"/>
  </integration>
</config>
""",
        encoding="utf-8",
    )

    assert run_config(load_config(config_path)) == 0

    text = out.read_text(encoding="utf-8")
    assert "sow,x_m,y_m,z_m" in text
    assert "tight" in text
