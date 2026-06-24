import math
import numpy as np
import pytest

from compass.ins.mechanization import (
    ECEFInertialState,
    IMURecord,
    attitude_from_ie_hpr_rfu,
    attitude_from_velocity_rfu,
    normal_gravity_ecef,
    read_imu_file,
    strapdown_step_ecef,
)
from compass.ins.coupling import (
    GNSSPVMeasurement,
    LooselyCoupledINS,
    TightlyCoupledINS,
    RangeMeasurement,
    INSFilterConfig,
    read_range_measurements_csv,
)


def test_mechanization_static_gravity_balance():
    position = np.array([6378137.0, 0.0, 0.0])
    velocity = np.zeros(3)
    gravity = normal_gravity_ecef(position)
    C_be = np.eye(3)
    state = ECEFInertialState(0.0, position.copy(), velocity.copy(), C_be, np.zeros(3), np.zeros(3))
    # Specific force opposite gravity should keep a stationary ECEF state nearly fixed.
    sample = IMURecord(1.0, np.zeros(3), -gravity, 1.0)
    propagated = strapdown_step_ecef(state, sample)
    assert np.linalg.norm(propagated.velocity) < 1.0e-3
    assert np.linalg.norm(propagated.position - position) < 1.0e-3


def test_read_imu_axis_order_converts_to_internal_rfu(tmp_path):
    imu_path = tmp_path / "imu.txt"
    imu_path.write_text("0.0 1 2 3 4 5 6\n0.1 1 2 3 4 5 6\n", encoding="utf-8")
    records = read_imu_file(imu_path, gyro_unit="dps", axis_order="gafrd")
    np.testing.assert_allclose(records[0].gyro, np.array([2.0, 1.0, -3.0]) * math.pi / 180.0)
    np.testing.assert_allclose(records[0].accel, np.array([5.0, 4.0, -6.0]))


def test_ie_zero_attitude_maps_rfu_to_local_axes():
    position = np.array([6378137.0, 0.0, 0.0])
    C_be = attitude_from_ie_hpr_rfu(position, 0.0, 0.0, 0.0)
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(C_be, expected, atol=1.0e-12)


def test_velocity_attitude_maps_forward_to_velocity():
    position = np.array([6378137.0, 0.0, 0.0])
    velocity = np.array([0.0, 10.0, 0.0])
    C_be = attitude_from_velocity_rfu(position, velocity)
    np.testing.assert_allclose(C_be @ np.array([0.0, 1.0, 0.0]), velocity / np.linalg.norm(velocity), atol=1.0e-12)
    np.testing.assert_allclose(C_be @ np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), atol=1.0e-12)


def test_loose_coupling_hard_reset_pv():
    state = ECEFInertialState(0.0, np.array([1.0, 2.0, 3.0]), np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
    filt = LooselyCoupledINS(state, INSFilterConfig(measurement_gate_sigma=1000.0, hard_reset_pv=True))
    measurement = GNSSPVMeasurement(
        0.0,
        np.array([10.0, 20.0, 30.0]),
        np.array([1.0, 2.0, 3.0]),
        np.full(3, 0.1),
        np.full(3, 0.2),
        1,
        1,
    )
    snapshot = filt.update_pv(measurement)
    assert snapshot.gnss_used == 1
    np.testing.assert_allclose(filt.state.position, measurement.position)
    np.testing.assert_allclose(filt.state.velocity, measurement.velocity)


def test_tight_coupling_range_update_reduces_position_error():
    true_position = np.array([6378137.0, 0.0, 0.0])
    initial_position = true_position + np.array([10.0, -4.0, 3.0])
    state = ECEFInertialState(0.0, initial_position, np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
    filt = TightlyCoupledINS(state, INSFilterConfig(measurement_gate_sigma=1000.0, hard_reset_pv=False))
    sats = [
        np.array([15600000.0, 21700000.0, 20100000.0]),
        np.array([18700000.0, -13400000.0, 23200000.0]),
        np.array([-17600000.0, 14100000.0, 21700000.0]),
        np.array([21100000.0, 7000000.0, -14100000.0]),
        np.array([-15000000.0, -21000000.0, 12000000.0]),
    ]
    measurements = [
        RangeMeasurement(0.0, sat, float(np.linalg.norm(true_position - sat)), 1.0)
        for sat in sats
    ]
    before = np.linalg.norm(filt.state.position - true_position)
    snapshot = filt.update_ranges(measurements)
    after = np.linalg.norm(filt.state.position - true_position)
    assert snapshot.gnss_used >= 4
    assert after < before


def test_tight_coupling_rejects_bad_range_measurement():
    true_position = np.array([6378137.0, 0.0, 0.0])
    initial_position = true_position + np.array([8.0, -3.0, 2.0])
    state = ECEFInertialState(0.0, initial_position, np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
    filt = TightlyCoupledINS(
        state,
        INSFilterConfig(
            measurement_gate_sigma=1000.0,
            hard_reset_pv=False,
            tight_range_gate_sigma=6.0,
            tight_auto_init_clock=True,
        ),
    )
    sats = [
        np.array([15600000.0, 21700000.0, 20100000.0]),
        np.array([18700000.0, -13400000.0, 23200000.0]),
        np.array([-17600000.0, 14100000.0, 21700000.0]),
        np.array([21100000.0, 7000000.0, -14100000.0]),
        np.array([-15000000.0, -21000000.0, 12000000.0]),
        np.array([21000000.0, -9000000.0, 18000000.0]),
    ]
    measurements = []
    for index, sat in enumerate(sats):
        pseudorange = float(np.linalg.norm(true_position - sat))
        if index == len(sats) - 1:
            pseudorange += 500.0
        measurements.append(RangeMeasurement(0.0, sat, pseudorange, 1.0))

    before = np.linalg.norm(filt.state.position - true_position)
    snapshot = filt.update_ranges(measurements)
    after = np.linalg.norm(filt.state.position - true_position)

    assert 4 <= snapshot.gnss_used < len(measurements)
    assert after < before


def test_tight_coupling_ls_blend_stabilizes_range_position():
    true_position = np.array([6378137.0, 0.0, 0.0])
    initial_position = true_position + np.array([30.0, -20.0, 10.0])
    state = ECEFInertialState(0.0, initial_position, np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
    filt = TightlyCoupledINS(
        state,
        INSFilterConfig(measurement_gate_sigma=1000.0, tight_position_ls_blend=1.0),
    )
    sats = [
        np.array([15600000.0, 21700000.0, 20100000.0]),
        np.array([18700000.0, -13400000.0, 23200000.0]),
        np.array([-17600000.0, 14100000.0, 21700000.0]),
        np.array([21100000.0, 7000000.0, -14100000.0]),
        np.array([-15000000.0, -21000000.0, 12000000.0]),
    ]
    measurements = [RangeMeasurement(0.0, sat, float(np.linalg.norm(true_position - sat)), 1.0) for sat in sats]

    before = np.linalg.norm(filt.state.position - true_position)
    snapshot = filt.update_ranges(measurements)
    after = np.linalg.norm(filt.state.position - true_position)

    assert snapshot.gnss_used == len(measurements)
    assert after < 1.0
    assert after < before


def test_tight_coupling_estimates_multisystem_isb_with_ls_blend():
    true_position = np.array([6378137.0, 0.0, 0.0])
    initial_position = true_position + np.array([25.0, -15.0, 8.0])
    state = ECEFInertialState(0.0, initial_position, np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
    filt = TightlyCoupledINS(
        state,
        INSFilterConfig(measurement_gate_sigma=1000.0, tight_position_ls_blend=1.0, tight_min_ranges=4),
    )
    sats = [
        ("G", np.array([15600000.0, 21700000.0, 20100000.0]), 0.0),
        ("G", np.array([18700000.0, -13400000.0, 23200000.0]), 0.0),
        ("G", np.array([-17600000.0, 14100000.0, 21700000.0]), 0.0),
        ("G", np.array([21100000.0, 7000000.0, -14100000.0]), 0.0),
        ("E", np.array([-15000000.0, -21000000.0, 12000000.0]), 30.0),
        ("E", np.array([21000000.0, -9000000.0, 18000000.0]), 30.0),
        ("C", np.array([23000000.0, 12000000.0, -9000000.0]), -45.0),
        ("C", np.array([-21000000.0, 16000000.0, 15000000.0]), -45.0),
    ]
    measurements = [
        RangeMeasurement(0.0, sat, float(np.linalg.norm(true_position - sat) + bias), 1.0, system=system)
        for system, sat, bias in sats
    ]

    snapshot = filt.update_ranges(measurements)
    after = np.linalg.norm(filt.state.position - true_position)

    assert snapshot.gnss_used == len(measurements)
    assert after < 1.0
    assert filt.isb_m[0] == pytest.approx(30.0, abs=1.0)
    assert filt.isb_m[1] == pytest.approx(-45.0, abs=1.0)


def test_read_range_measurements_csv_groups_epochs(tmp_path):
    path = tmp_path / "ranges.csv"
    path.write_text(
        "\n".join(
            [
                "sow,sat_x_m,sat_y_m,sat_z_m,pseudorange_m,variance_m2,sat_vx_mps,sat_vy_mps,sat_vz_mps,range_rate_mps,range_rate_variance_m2",
                "1,2,3,4,5,9,0.1,0.2,0.3,-1,0.04",
                "1,3,4,5,6,16,,,,,",
                "2,4,5,6,7,25,,,,,",
            ]
        ),
        encoding="utf-8",
    )

    epochs = read_range_measurements_csv(path)

    assert tuple(epochs) == (1.0, 2.0)
    assert len(epochs[1.0]) == 2
    assert epochs[1.0][0].pseudorange == 5.0
    np.testing.assert_allclose(epochs[1.0][0].sat_velocity, np.array([0.1, 0.2, 0.3]))
    assert epochs[1.0][0].range_rate == -1.0
