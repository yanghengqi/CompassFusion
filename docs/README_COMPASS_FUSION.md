# CompassFusion

CompassFusion is the standalone GNSS/INS processing entry in this workspace. It wraps the existing GNSS, INS mechanization, loose-coupling and prototype tight-coupling code behind an XML configuration file plus lightweight command-line overrides.

## Current Scope

Implemented:

- INS ECEF mechanization.
- GNSS/INS loose coupling from GNSS position/velocity CSV or GREAT/Inertial Explorer P/V text.
- Prototype tightly coupled INS update from prebuilt satellite range CSV.
- Real RINEX SPP export to INS P/V CSV and tight range CSV.
- Real SPP-INS and PPP-INS quick regression plots.
- XML-driven processing entry: `run_compass_fusion.py`.
- GREAT-MSF 2021-10-13 example configuration.
- Regression test script that runs loose and tight cases, writes statistics, and generates plots.

In progress:

- Full RINEX-to-tight frontend.
- SPP-INS, PPK-INS and PPP-INS end-to-end test matrix.
- Static GNSS, dynamic GNSS and fused GNSS/INS release reports.

## Quick Start

Validate the example configuration:

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' run_compass_fusion.py --config configs\compass_fusion_211013.xml --check --verbose
```

Run the GREAT loose-coupling example:

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' run_compass_fusion.py --config configs\compass_fusion_211013.xml
```

Run the convenience PowerShell wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_compass_fusion_211013.ps1
```

Use diagnostic epoch attitude:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_compass_fusion_211013.ps1 -EpochAttitude
```

## Regression Tests And Plots

Run quick loose and tight regression tests:

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' scripts\run_compass_fusion_tests.py --out-dir results\compass_fusion_tests_quick
```

Run quick real RINEX SPP-INS tests:

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' scripts\run_compass_fusion_tests.py --real-spp --out-dir results\compass_fusion_real_spp_quick
```

Run quick real PPP-INS tests:

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' scripts\run_compass_fusion_tests.py --real-ppp --out-dir results\compass_fusion_real_ppp_quick
```

Run the release quick suite:

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' scripts\run_compass_fusion_tests.py --real-spp --real-ppk --real-ppp --out-dir results\compass_fusion_release_candidate
```

Run the full GREAT loose segment plus tight synthetic test:

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' scripts\run_compass_fusion_tests.py --full --out-dir results\compass_fusion_tests_full
```

Run the full real RINEX SPP-INS test:

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' scripts\run_compass_fusion_tests.py --real-spp --full --out-dir results\compass_fusion_real_spp_full
```

Outputs:

- `results/compass_fusion_tests_quick/summary.csv`
- `results/compass_fusion_tests_quick/plots/loose_great_error_timeseries.png`
- `results/compass_fusion_tests_quick/plots/loose_great_error_cdf.png`
- `results/compass_fusion_tests_quick/plots/tight_synthetic_error_timeseries.png`
- `results/compass_fusion_tests_quick/plots/tight_synthetic_error_cdf.png`

Latest quick regression in this workspace:

| Case | Mode | Matched | RMS (m) | P95 (m) | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| loose_great | loose | 60 | 0.0162 | 0.0167 | GREAT-MSF short segment, IE P/V input |
| tight_synthetic | tight | 31 | 0.0906 | 0.1872 | Synthetic satellite pseudorange smoke test |

The tight synthetic case is a deterministic pipeline test. It confirms tight-coupling scheduling, range CSV parsing, filtering and plotting. It is not yet a real RINEX/PPP/RTK tightly coupled benchmark.

Latest real RINEX SPP-INS quick regression:

| Case | Mode | Matched | RMS (m) | P95 (m) | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| real_spp_gec_loose | loose | 60 | 2.1141 | 3.8825 | SEPT2860 RINEX SPP P/V + INS |
| real_spp_gec_tight | tight | 60 | 15.4006 | 16.9186 | Real pseudorange tight input with LS stabilizer and ISB state; multi-system code model still limiting |
| real_spp_g_loose | loose | 60 | 3.6440 | 5.5963 | GPS-only SPP P/V + INS |
| real_spp_g_tight | tight | 60 | 3.9552 | 5.8070 | GPS-only real pseudorange tight input |
| real_ppp_gec_loose | loose | 60 | 1.3747 | 1.4040 | Existing SEPT2860 PPP multipass P/V + INS |
| real_ppk_gec_loose | loose | 60 | 0.5695 | 0.7688 | GREAT SEPT/R293 PPK P/V + INS |

Latest release candidate quick results:

| Case | Mode | Matched | RMS (m) | P95 (m) |
| --- | --- | ---: | ---: | ---: |
| loose_great | loose | 60 | 0.0162 | 0.0167 |
| tight_synthetic | tight | 31 | 0.0906 | 0.1872 |
| real_spp_gec_loose | loose | 60 | 2.1141 | 3.8825 |
| real_spp_gec_tight | tight | 60 | 15.4006 | 16.9186 |
| real_ppp_gec_loose | loose | 60 | 1.3747 | 1.4040 |
| real_ppk_gec_loose | loose | 60 | 0.5695 | 0.7688 |

Release candidate outputs:

- `results/compass_fusion_release_candidate/summary.csv`
- `results/compass_fusion_release_candidate/plots/`

## Release Plots

SPP-INS loose:

![SPP-INS loose error](results/compass_fusion_release_candidate/plots/real_spp_gec_loose_error_timeseries.png)

PPK-INS loose:

![PPK-INS loose error](results/compass_fusion_release_candidate/plots/real_ppk_gec_loose_error_timeseries.png)

PPP-INS loose:

![PPP-INS loose error](results/compass_fusion_release_candidate/plots/real_ppp_gec_loose_error_timeseries.png)

SPP-INS tight experimental:

![SPP-INS tight error](results/compass_fusion_release_candidate/plots/real_spp_gec_tight_error_timeseries.png)

Latest real RINEX SPP-INS full/continuous diagnostics:

| Case | Mode | Matched | Median (m) | RMS (m) | P95 (m) | Max (m) | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| real_spp_gec_loose_continuous | loose | 3274 | 1.5127 | 26.7408 | 20.3462 | 576.2322 | Runs, affected by SPP outliers |
| real_spp_gec_loose_cut | loose | 3729 | 1.6545 | 1269.5084 | 60.2537 | 21635.8337 | Runs, affected by SPP outage/free INS drift |
| real_spp_gec_tight_continuous | tight | 3274 | 3399936.4549 | 11812808.2181 | 26161942.7672 | 30860112.7154 | Fails precision gate |
| real_spp_gec_tight_cut | tight | 3729 | 6047959.1081 | 17956325.6293 | 41307428.5026 | 51549462.8313 | Fails precision gate |

Real full-test plots are in:

- `results/compass_fusion_real_spp_full/plots/`
- `results/compass_fusion_real_spp_full/real_spp_gec/`

The regression script also writes `*_updated` rows. These rows measure epochs where a GNSS update was actually used (`gnss_used > 0`), which is useful when full data contain SPP/PPP outages. Full-trajectory rows intentionally include outage/free-INS drift.

Important current limitation: real tight coupling is stable for the GPS-only SPP quick test, and multi-system GEC no longer diverges in the short test after range-LS stabilization and ISB state support. It is still not final publication-grade for full multi-system operation. Remaining limiting items include better code-bias handling, stronger code outlier control, carrier-phase/ambiguity support, and an outage strategy. Treat the real tight results above as an engineering baseline.

## Release Baseline

This workspace is suitable as a CompassFusion engineering release baseline with these claims:

- GREAT/IE loose-coupling example is stable and reproducible.
- SPP-INS loose coupling runs on real RINEX-derived SPP P/V.
- PPP-INS loose coupling runs on existing SEPT2860 PPP P/V.
- PPK-INS loose coupling runs on GREAT SEPT/R293 short-baseline PPK P/V.
- GPS-only SPP-INS tight coupling runs on real RINEX pseudorange input.
- Multi-system SPP-INS tight coupling is available as an experimental mode.
- Test scripts generate CSV summaries and PNG plots for release reports.

Not claimed yet:

- Full PPP/RTK carrier-phase tight coupling.
- Integer ambiguity tight coupling.
- Production-grade multi-system tight code-bias calibration.
- Robust long-outage INS bridging.

## Tight Range CSV Prototype

`mode=tight` currently expects prebuilt satellite range measurements:

```csv
sow,sat_x_m,sat_y_m,sat_z_m,pseudorange_m,variance_m2
```

Optional Doppler/range-rate fields:

```csv
sat_vx_mps,sat_vy_mps,sat_vz_mps,range_rate_mps,range_rate_variance_m2
```

Example XML:

```xml
<inputs>
  <imu>path/to/imu.txt</imu>
  <gnss format="csv">path/to/initial_pv.csv</gnss>
  <ranges format="csv">path/to/ranges.csv</ranges>
</inputs>
```

## Test Matrix Target

Planned release matrix:

| Category | GNSS Only | Loose Coupling | Tight Coupling |
| --- | --- | --- | --- |
| Static GNSS | SPP, PPP, PPK | PPP-INS static, PPK-INS static | PPP-INS tight static |
| Dynamic GNSS | SPP, PPP, PPK | SPP-INS, PPP-INS, PPK-INS | SPP-INS tight, PPP-INS tight, PPK-INS tight |
| Fusion Report | Error statistics and plots | Error statistics and plots | Error statistics and plots |

## Development Verification

Run all unit and smoke tests:

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' -m pytest
```

Current test count after adding loose/tight regression coverage:

```text
42 passed
```
