# Core Code Manifest

## Package Name
CompassFusion_core_code_v0_1

## Included
- Unified front-end: `src/run_compass_fusion.py`
- GNSS engines: SPP / PPP / PPK modules under `src/compass/gnss/`
- INS engines: mechanization / loose coupling / tight coupling under `src/compass/ins/`
- RINEX and native input support under `src/compass/io/`
- Config template: `configs/compass_fusion_211013.xml`
- Test and data-preparation scripts under `scripts/`
- Regression tests under `tests/`
- Release documentation under `docs/`

## Excluded
- Raw GNSS/IMU datasets
- Large generated result folders
- Python bytecode and cache folders
- Temporary profiling outputs
## Included Data Examples
- `data_examples/great_msf_20211013/GNSS/`: rover/base RINEX observation files and broadcast navigation file.
- `data_examples/great_msf_20211013/IMU/`: real IMU sample file.
- `data_examples/great_msf_20211013/groundtruth/`: GNSS and GNSS/INS reference trajectories.
- `data_examples/great_msf_20211013/model/`: small model files needed by typical navigation tests.- `data_examples/great_msf_20211013/products/sp3/`: precise orbit product.
- `data_examples/great_msf_20211013/products/clk/`: precise clock product.
- `data_examples/great_msf_20211013/products/bia/`: OSB and ORBEX auxiliary products.
- `data_examples/great_msf_20211013/products/erp/`: Earth rotation product.
- data_examples/great_msf_20211013/products/dcb/: CODE monthly DCB fallback products.
- scripts/run_ppp_great_msf_example.ps1: PPP example runner using included products.

