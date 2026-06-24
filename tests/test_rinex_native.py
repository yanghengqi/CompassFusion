from io import StringIO

from compass.core.gnss_types import GNSSRawObservation, SatelliteObservation
from compass.io.rinex_native import RINEXNativeReader


def test_long_satellite_line_does_not_consume_next_satellite():
    first = "G01" + " " * (16 * 22) + "\n"
    second = "G02" + " " * (16 * 24) + "\n"
    stream = StringIO(second)

    record = RINEXNativeReader._read_sat_record_v3(stream, first, 24)

    assert record.startswith("G01")
    assert stream.readline().startswith("G02")


def test_standard_continuation_line_is_merged():
    first = "G01" + "1" * (16 * 4) + "\n"
    continuation = "   " + "2" * (16 * 2) + "\n"
    next_epoch = "> 2021 07 15 00 00 30.0000000  0  1\n"
    stream = StringIO(continuation + next_epoch)

    record = RINEXNativeReader._read_sat_record_v3(stream, first, 6)

    assert record == "G01" + "1" * (16 * 4) + "2" * (16 * 2)
    assert stream.readline().startswith(">")

def test_pots_epochs_remain_synchronized_across_event_records():
    reader = RINEXNativeReader()

    epochs = reader.read_obs("data/obs/pots1960.21o", max_epochs=31)

    assert len(epochs) == 31
    assert [epoch.timestamp for epoch in epochs] == [345600.0 + 30.0 * i for i in range(31)]
    assert min(len(epoch.observations) for epoch in epochs) >= 40
    assert reader.obs_receiver_antenna == "JAVRINGANT_G5T"
    assert reader.obs_antenna_delta_enu.tolist() == [0.0, 0.0, 0.1206]

def _doppler_epochs(doppler_sign):
    epochs = []
    for index in range(2):
        satellites = []
        for prn in range(1, 5):
            phase = 1000.0 * prn + 100.0 * index
            doppler = doppler_sign * 100.0
            satellites.append(SatelliteObservation(
                prn, "G", 22_000_000.0, 22_000_000.0, phase, phase,
                doppler_L1=doppler, doppler_L2=doppler,
                raw_observations={"D1C": (doppler, 0, 45.0)},
            ))
        epochs.append(GNSSRawObservation(float(index), 2200, satellites))
    return epochs


def test_receiver_specific_doppler_sign_is_normalized():
    epochs = _doppler_epochs(1.0)

    reversed_sign = RINEXNativeReader._normalize_doppler_convention(epochs)

    assert reversed_sign
    assert epochs[0].observations[0].doppler_L1 == -100.0
    assert epochs[0].observations[0].raw_observations["D1C"][0] == -100.0


def test_standard_doppler_sign_is_preserved():
    epochs = _doppler_epochs(-1.0)

    reversed_sign = RINEXNativeReader._normalize_doppler_convention(epochs)

    assert not reversed_sign
    assert epochs[0].observations[0].doppler_L1 == -100.0
