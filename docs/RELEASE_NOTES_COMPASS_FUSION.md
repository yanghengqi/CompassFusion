# CompassFusion v0.1 Engineering Release Candidate

## Release Status

This release candidate is suitable for engineering evaluation and demonstration.

Stable release claims:

- GREAT-MSF XML-driven GNSS/INS loose-coupling example.
- Real RINEX SPP export to GNSS P/V and pseudorange range CSV.
- Real SPP-INS loose-coupling quick test.
- Real PPK-INS loose-coupling quick test using GREAT SEPT/R293.
- Real PPP-INS loose-coupling quick test using existing SEPT2860 PPP multipass output.
- GPS-only SPP-INS pseudorange tight-coupling quick test.
- Unified regression script, CSV summary, and PNG plots.

Experimental:

- Multi-system SPP-INS pseudorange tight coupling.
- Full-duration tests through GNSS outages.

Not included in v0.1:

- PPP carrier-phase tight coupling.
- RTK/PPK carrier-phase tight coupling with ambiguities in the INS filter.
- Production multi-system code-bias calibration for tight coupling.
- Robust long-outage inertial bridging.

## Main Command

```powershell
& 'D:\annaconda\envs\BraVL\python.exe' scripts\run_compass_fusion_tests.py --real-spp --real-ppk --real-ppp --out-dir results\compass_fusion_release_candidate
```

## Release Candidate Quick Results

| Case | Mode | RMS (m) | P95 (m) |
| --- | --- | ---: | ---: |
| GREAT IE P/V INS | loose | 0.0162 | 0.0167 |
| Synthetic pseudorange INS | tight | 0.0906 | 0.1872 |
| SPP-INS | loose | 2.1141 | 3.8825 |
| SPP-INS | tight experimental | 15.4006 | 16.9186 |
| PPP-INS | loose | 1.3747 | 1.4040 |
| PPK-INS | loose | 0.5695 | 0.7688 |

## Verification

```text
42 passed
```

## Output Artifacts

- `results/compass_fusion_release_candidate/summary.csv`
- `results/compass_fusion_release_candidate/plots/`
- `README_COMPASS_FUSION.md`
