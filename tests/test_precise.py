import numpy as np

from compass.gnss.precise import PreciseProducts


def _products():
    products = PreciseProducts(clock_edge_tolerance=1.0)
    products.orbits["G01"] = [
        (100.0, np.array([20_000_000.0, 10_000_000.0, 5_000_000.0]), 1.0e-6),
        (400.0, np.array([20_000_300.0, 10_000_000.0, 5_000_000.0]), 2.0e-6),
    ]
    products.clocks["G01"] = [(100.0, 1.0e-6), (130.0, 1.1e-6)]
    return products


def test_state_full_clamps_clock_just_before_product_start():
    state = _products().state_full("G", 1, 99.92)

    assert state is not None
    assert np.isfinite(state[2])


def test_state_full_rejects_time_beyond_clock_edge_tolerance():
    assert _products().state_full("G", 1, 98.9) is None


def test_linear_still_rejects_large_internal_gap():
    rows = [(100.0, 1.0), (300.0, 2.0)]

    assert PreciseProducts._linear(rows, 200.0, 1, max_gap=120.0, edge_tolerance=1.0) is None