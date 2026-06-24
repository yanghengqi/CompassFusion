# CompassFusion

CompassFusion is a standalone GNSS/INS processing toolkit built from the `spp_standalone` engineering workspace. This repository contains the core Python code, configuration files, tests, documentation, and one real GREAT/MSF sample dataset for demonstration.

## Repository Contents

- `src/`: core source code and command-line entry points.
- `src/run_compass_fusion.py`: unified XML front-end for INS mechanization, loose coupling, and tight-coupling experiments.
- `src/compass/gnss/`: SPP, PPP, PPK, satellite models, precise products, and bias models.
- `src/compass/ins/`: IMU mechanization, loose coupling, and tight coupling.
- `configs/`: XML configuration templates.
- `scripts/`: data export, batch tests, comparison, and diagnostic scripts.
- `tests/`: regression tests.
- `data_examples/`: bundled real navigation sample data and product files.
- `docs/`: engineering README and release notes.

## Bundled Sample Data

The bundled sample dataset is located at:

- `data_examples/great_msf_20211013/`

It includes:

- rover/base RINEX observation files
- broadcast navigation file
- IMU raw data
- GNSS reference trajectory
- GNSS/INS reference trajectory
- SP3 precise orbit
- CLK precise clock
- OSB/OBX bias and attitude products
- ERP/DCB products
- ATX/EOP/ocean-loading/tide/JPL model files

See `data_examples/DATASETS.md` for the detailed data description.

## Recommended Python Environment

The current verified Python environment is:

```powershell
D:\annaconda\envs\BraVL\python.exe
```

## Quick Start

Run the included CompassFusion example from the repository root:

```powershell
$env:PYTHONPATH = "$PWD\src"
& 'D:\annaconda\envs\BraVL\python.exe' src\run_compass_fusion.py --config configs\compass_fusion_great_msf_example.xml
```

Run the PPP example with bundled precise products:

```powershell
scripts\run_ppp_great_msf_example.ps1
```

Run tests:

```powershell
$env:PYTHONPATH = "$PWD\src"
& 'D:\annaconda\envs\BraVL\python.exe' -m pytest tests
```

## Current Scope

Stable engineering functions currently include GNSS processing, INS mechanization, loose GNSS/INS coupling, and pseudorange-level tight-coupling experiments. Full carrier-phase PPP/RTK tight coupling and production-grade ambiguity fixing are still future enhancement items.

## License

MIT License. See `LICENSE`.
