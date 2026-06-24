from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_compass_fusion_tests import _read_csv_truth, _run_case, _write_tight_synthetic_case


def test_compass_fusion_regression_script_runs_tight_synthetic(tmp_path):
    config, truth = _write_tight_synthetic_case(tmp_path / "tight", duration_s=3)
    row = _run_case("tight_unit", "tight", config, _read_csv_truth(truth), tmp_path / "plots")

    assert row["case"] == "tight_unit"
    assert row["mode"] == "tight"
    assert row["matched"] == 4.0
    assert row["rms_m"] < 1.0
    assert (tmp_path / "plots" / "tight_unit_error_timeseries.png").exists()
