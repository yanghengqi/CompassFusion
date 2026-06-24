import numpy as np

from compass.core.constants import CLIGHT
from compass.core.gnss_types import GNSSRawObservation, SatelliteObservation
from compass.core.transforms import ecef2llh
from compass.gnss.ppp import PPPConfig, PPPKalmanFilter
from compass.gnss.ppk import PPKConfig, PPKKalmanFilter
from compass.gnss.bias_osb import OsbCalibration
from compass.gnss.precise import PreciseProducts
from compass.gnss.ppp_models import OceanLoading
from compass.gnss.trajectory_smoother import RTSSnapshot, covariance_intersection, rts_smooth, trajectory_multipass
from compass.gnss.lambda_ar import lambda_algorithm


def _scenario():
    week, sow0 = 2200, 100000.0
    true = np.array([6378137.0, 0.0, 0.0])
    directions = np.array([
        [0.80, 0.60, 0.00], [0.75, -0.55, 0.37], [0.70, 0.10, -0.71],
        [0.62, -0.20, -0.76], [0.67, 0.70, 0.25], [0.58, -0.75, 0.32],
    ])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    raw_sats = [true + 2.2e7 * d for d in directions]
    products = PreciseProducts(max_clock_gap=120.0)
    for prn, sat in enumerate(raw_sats, 1):
        products.orbits[f"G{prn:02d}"] = [(week * 604800 + sow0 + dt, sat.copy(), 0.0) for dt in range(-60, 181, 30)]
        products.clocks[f"G{prn:02d}"] = [(week * 604800 + sow0 + dt, 0.0) for dt in range(-60, 181, 30)]
    config = PPPConfig(elevation_mask_deg=5.0, jerk_noise_mps3=0.01, code_sigma_m=0.3)
    flt = PPPKalmanFilter(products, config=config)
    epochs = []
    llh = ecef2llh(true)
    for k in range(12):
        observations = []
        for prn, raw in enumerate(raw_sats, 1):
            travel = 2.2e7 / CLIGHT
            sat = flt._rotate(raw, np.zeros(3), travel)[0]
            travel = np.linalg.norm(sat - true) / CLIGHT
            sat = flt._rotate(raw, np.zeros(3), travel)[0]
            rho = np.linalg.norm(sat - true)
            _, el = flt._azel(sat, true, llh)
            hydro, wet_map = flt._troposphere(llh, el, week * 604800 + sow0 + k * 10.0)
            common = rho + 75.0 + hydro + wet_map * 0.15
            ambiguity = 4.0 + prn
            f1, f2 = 1575.42e6, 1227.60e6
            observations.append(SatelliteObservation(
                prn, "G", common, common,
                (common + ambiguity) * f1 / CLIGHT,
                (common + ambiguity) * f2 / CLIGHT,
            ))
        epochs.append(GNSSRawObservation(sow0 + k * 10.0, week, observations, approx_position=true + [20.0, -10.0, 5.0]))
    return true, flt, epochs


def test_ppp_converges_and_accepts_navigation_prior():
    true, flt, epochs = _scenario()
    sol = None
    for epoch in epochs:
        sol = flt.process(epoch)
    assert sol is not None and sol.status == 1
    assert np.linalg.norm(sol.position - true) < 0.5
    flt.apply_navigation_prior(true, np.eye(3) * 0.01)
    assert np.linalg.norm(flt.x[:3] - true) < 0.2

def test_doppler_combination_uses_rinex_sign_convention():
    flt = PPPKalmanFilter(PreciseProducts())
    obs = SatelliteObservation(
        1, "G", 22_000_000.0, 22_000_000.0,
        115_000_000.0, 90_000_000.0,
        doppler_L1=-1200.0, doppler_L2=-900.0,
    )

    combination = flt._combinations(obs)

    assert combination is not None
    assert combination[4] > 0.0


def test_explicit_approx_bypasses_spp_position_recovery():
    true = np.array([3800689.375, 882077.650, 5028791.485])
    wrong_spp = true + np.array([300.0, 0.0, 0.0])
    flt = PPPKalmanFilter(PreciseProducts(), nav=[object()])
    flt.spp.solve = lambda epoch, nav, seed: (wrong_spp.copy(), -90_000.0, 1)
    epoch = GNSSRawObservation(345600.0, 2166, [], approx_position=wrong_spp)

    flt.process(epoch, true)

    assert np.array_equal(flt.x[:3], true)

def test_static_mode_accumulates_position_without_motion_states():
    true, flt, epochs = _scenario()
    flt.config.static_mode = True
    first_position_variance = None

    for epoch in epochs:
        solution = flt.process(epoch)
        if first_position_variance is None:
            first_position_variance = np.trace(flt.P[:3, :3])

    assert solution.status == 1
    assert np.linalg.norm(solution.position - true) < 0.5
    assert np.array_equal(flt.x[3:9], np.zeros(6))
    assert np.trace(flt.P[:3, :3]) < first_position_variance


def test_reverse_prediction_uses_negative_time_direction_with_positive_covariance():
    flt=PPPKalmanFilter(PreciseProducts(),config=PPPConfig(time_direction=-1))
    flt.x=np.zeros(flt.BASE_NX);flt.x[3]=2.0
    flt.P=np.eye(flt.BASE_NX);flt.last_time=100.0
    assert flt._predict(98.0)
    assert np.isclose(flt.x[0],-4.0)
    assert np.all(np.diag(flt.P)>=0.0)


def test_rts_smoother_reduces_earlier_uncertainty_without_crossing_failures():
    eye=np.eye(2)
    records=[
        RTSSnapshot(np.array([0.0,1.0]),eye.copy(),None,None,None,True,False),
        RTSSnapshot(np.array([1.2,1.0]),eye*.5,np.array([1.0,1.0]),eye*1.5,eye,True,True),
        RTSSnapshot(np.array([9.0,0.0]),eye.copy(),None,None,None,False,False),
    ]
    states,covariances=rts_smooth(records)
    assert states[0][0] > 0.0
    assert np.trace(covariances[0]) < np.trace(records[0].filtered_covariance)
    assert np.array_equal(states[2],records[2].filtered_state)


def test_covariance_intersection_stays_between_forward_and_reverse():
    state,covariance=covariance_intersection(np.array([0.0]),np.array([[1.0]]),np.array([2.0]),np.array([[1.0]]))
    assert np.allclose(state,[1.0])
    assert np.allclose(covariance,[[1.0]])


def test_lambda_candidate_matches_correlated_brute_force_solution():
    ambiguities=np.array([1.96621556,-0.54480518])
    covariance=np.array([[0.31942696,0.17116802],[0.17116802,0.39970291]])
    fixed,_,ok=lambda_algorithm(ambiguities,covariance,2)
    assert ok
    assert np.array_equal(fixed[:,0],[2.0,-1.0])


def test_trajectory_multipass_preserves_failed_arc_boundary():
    states=[np.zeros(9),np.ones(9),np.full(9,50.0)]
    covariances=[np.eye(9) for _ in states]
    smoothed,_=trajectory_multipass([0.0,1.0,2.0],states,covariances,[True,True,False],jerk_noise=0.1)
    assert np.linalg.norm(smoothed[0]-states[1])<np.linalg.norm(states[0]-states[1])
    assert np.array_equal(smoothed[2],states[2])

def _raw_dual_frequency(system, sat_id, first, second):
    raw = {
        "C" + first: (22_000_000.0, 0, 45.0),
        "L" + first: (115_000_000.0, 0, 45.0),
        "C" + second: (22_000_003.0, 0, 45.0),
        "L" + second: (90_000_000.0, 0, 45.0),
    }
    return SatelliteObservation(
        sat_id, system, 22_000_000.0, 22_000_003.0,
        115_000_000.0, 90_000_000.0, raw_observations=raw,
    )


def test_great_signal_mapping_uses_galileo_e5a():
    obs = _raw_dual_frequency("E", 2, "1X", "5X")
    result = PPPKalmanFilter(PreciseProducts())._combinations(obs)

    assert result is not None
    assert result[6:9] == (1176.45e6, "E01", "E05")


def test_great_signal_mapping_uses_bds3_b3i():
    obs = _raw_dual_frequency("C", 19, "2I", "6I")
    result = PPPKalmanFilter(PreciseProducts())._combinations(obs)

    assert result is not None
    assert result[6:9] == (1268.52e6, "C02", "C06")

def test_ocean_loading_reads_pots_blq_and_returns_finite_displacement():
    loading = OceanLoading.from_file(
        "conference/compass-master/data/2021/196/daily/ocnload.blq", "POTS"
    )
    displacement = loading.displacement(
        np.array([3800689.375, 882077.650, 5028791.485]), 2166*604800.0+345600.0
    )

    assert loading.coefficients.shape == (6, 11)
    assert np.all(np.isfinite(displacement))
    assert np.linalg.norm(displacement) < 0.1

def test_ppp_ar_fixes_consistent_single_difference_ambiguities():
    config = PPPConfig(
        ambiguity_resolution=True, ar_min_epochs=10, ar_min_satellites=5
    )
    osb = OsbCalibration(code_bias_m={("G01", "C1C"): 0.0})
    flt = PPPKalmanFilter(PreciseProducts(), config=config, osb=osb)
    flt.x = np.zeros(flt.BASE_NX + 5)
    flt.P = np.eye(flt.BASE_NX + 5) * 1.0e-4
    f1, f2 = 1575.42e6, 1227.60e6
    wide_lane = CLIGHT / (f1 - f2)
    lambda1 = CLIGHT / f1
    narrow_lane = CLIGHT / (f1 + f2)
    alpha = f1*f1 / (f1*f1 - f2*f2)

    for index in range(5):
        key = ("G", index + 1)
        flt.ambiguity_index[key] = flt.BASE_NX + index
        flt.mw_count[key] = 100
        flt.mw_m2[key] = 0.01
        flt.mw_mean[key] = (100 + index) * wide_lane
        flt.ambiguity_frequency[key] = (f1, f2)
        flt.last_elevation[key] = np.radians(40 + index)
        flt.x[flt.BASE_NX + index] = 10.0 + alpha*lambda1*index + narrow_lane*2*index

    fixed, count, ratio = flt._try_fix_ambiguities()

    assert fixed == 1
    assert count == 4
    assert ratio >= config.ar_ratio_threshold
    flt.last_time = 100.0
    for key in flt.ambiguity_index:
        flt.last_seen[key] = 100.0
    held, held_count, held_ratio = flt._try_fix_ambiguities()
    assert held == 1
    assert held_count == 4
    assert np.isinf(held_ratio)
    slipped = ("G", 5)
    flt._rebase_held_ambiguities(slipped)
    assert all(slipped not in pair for pair in flt.held_ambiguities)
    assert len(flt.held_ambiguities) == 3


def test_ppp_ar_hold_reselects_visible_reference():
    config = PPPConfig(ambiguity_resolution=True, ar_min_satellites=5)
    osb = OsbCalibration(code_bias_m={("G01", "C1C"): 0.0})
    flt = PPPKalmanFilter(PreciseProducts(), config=config, osb=osb)
    flt.x = np.zeros(flt.BASE_NX + 6)
    flt.P = np.eye(flt.BASE_NX + 6)
    flt.last_time = 100.0
    reference = ("G", 1)
    for index in range(6):
        key = ("G", index + 1)
        flt.ambiguity_index[key] = flt.BASE_NX + index
        flt.last_seen[key] = flt.last_time
        flt.last_elevation[key] = np.radians(50 - index)
    flt.held_ambiguities = {
        (("G", index), reference): float(index - 1) for index in range(2, 7)
    }
    flt.last_seen[reference] -= 30.0
    held, held_count, held_ratio = flt._try_fix_ambiguities()

    assert held == 1
    assert held_count == 4
    assert np.isinf(held_ratio)


def test_static_ar_position_hold_uses_recent_fixed_median():
    config = PPPConfig(
        ambiguity_resolution=True, static_mode=True,
        static_hold_min_fixed_epochs=3, static_hold_window_epochs=3,
    )
    osb = OsbCalibration(code_bias_m={("G01", "C1C"): 0.0})
    flt = PPPKalmanFilter(PreciseProducts(), config=config, osb=osb)
    flt.x = np.zeros(flt.BASE_NX)
    flt.P = np.eye(flt.BASE_NX)
    for position in ([1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [20.0, 2.2, 3.2]):
        flt.x[:3] = position
        flt._apply_static_position_hold(1)

    assert np.allclose(flt.static_position_anchor, [1.1, 2.1, 3.1])


def test_ppk_wide_lane_conditioning_does_not_mutate_float_state():
    config = PPKConfig(wide_lane_min_epochs=2)
    flt = PPKKalmanFilter(PreciseProducts(), np.zeros(3), config)
    flt.x = np.zeros(8)
    flt.P = np.eye(8)
    flt.reference = 1
    flt.ambiguity_index = {(1, 2): 6, (2, 2): 7}
    flt.mw_count = {1: 10, 2: 10}
    flt.mw_mean = {1: 0.0, 2: 5.05}
    flt.mw_m2 = {1: 0.01, 2: 0.01}
    state_before, covariance_before = flt.x.copy(), flt.P.copy()

    conditioned_state, conditioned_covariance = flt._apply_wide_lane_constraints(1)

    assert np.array_equal(flt.x, state_before)
    assert np.array_equal(flt.P, covariance_before)
    assert abs((conditioned_state[6] - conditioned_state[7]) - 5.0) < 0.02
    assert not np.array_equal(conditioned_covariance, covariance_before)


def test_ppk_confirmed_hold_feedback_only_locks_ambiguities():
    config = PPKConfig(
        ambiguity_strategy="all", ambiguity_min_epochs=1,
        ambiguity_hold_epochs=1, ambiguity_hold_feedback=True,
    )
    flt = PPKKalmanFilter(PreciseProducts(), np.zeros(3), config)
    flt.x = np.zeros(14)
    flt.P = np.eye(14)
    flt.reference = 1
    for offset, sat in enumerate(range(2, 6)):
        flt.ambiguity_index[(1, sat)] = 6 + 2 * offset
        flt.ambiguity_index[(2, sat)] = 7 + 2 * offset
        flt.ambiguity_age[(1, sat)] = flt.ambiguity_age[(2, sat)] = 10
        flt.held_integers[(1, sat)] = sat
        flt.held_integers[(2, sat)] = -sat
    position_before = flt.x[:6].copy()
    covariance_before = flt.P[:6, :6].copy()

    fixed, _, _, _ = flt._fix_ambiguities()

    assert fixed == 1
    assert np.array_equal(flt.x[:6], position_before)
    assert np.array_equal(flt.P[:6, :6], covariance_before)
    for sat in range(2, 6):
        assert flt.x[flt.ambiguity_index[(1, sat)]] == sat
        assert flt.x[flt.ambiguity_index[(2, sat)]] == -sat
