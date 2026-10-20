# PINN-WSGG-Pressure

> A bundled release containing the Python, OpenFOAM, and Fluent-UDF implementations of the pressure-dependent PINN-WSGG model for H<sub>2</sub>O-CO<sub>2</sub>-CO participating-gas mixtures.

---

## Overview

This directory collects three aligned packages for the same pressure-dependent PINN-WSGG model:

- `PINN-WSGG_Python`: Python implementation with 1D RTE testing against LBL data
- `PINN-WSGG_OpenFOAM`: OpenFOAM implementation with a shipped `thermoFoam` example case
- `PINN-WSGG_UDF`: Fluent compiled UDF implementation

Each subdirectory is self-contained and includes its own `README.md` and `LICENSE`.

---

## Repository Structure
```text
PINN-WSGG-Pressure/
├── PINN-WSGG_Python/         # Python model and test case
├── PINN-WSGG_OpenFOAM/       # OpenFOAM source and example case
└── PINN-WSGG_UDF/            # Fluent compiled UDF
```

---

## Usage

- For Python usage, see `PINN-WSGG_Python/README.md`
- For OpenFOAM usage, see `PINN-WSGG_OpenFOAM/README.md`
- For Fluent UDF usage, see `PINN-WSGG_UDF/README.md`

---

## Cite as

If you use this work in your research, please cite as follows,

```bibtex
@article{Wang2027PressurePINNWSGG,
  title = {Pressure-dependent spectral-moment-constrained physics-informed neural-network {WSGG} model for multi-species gas mixtures},
  author = {Wang, Jiaxin and Chen, Wei and Ren, Tao and Zhao, Changying},
  year = {2027},
  journal = {International Journal of Heat and Mass Transfer},
  volume = {273},
  pages = {129746},
  issn = {0017-9310},
  doi = {10.1016/j.ijheatmasstransfer.2026.129746}
}
```

J. Wang, W. Chen, T. Ren, and C. Zhao, “Pressure-dependent spectral-moment-constrained physics-informed neural-network WSGG model for multi-species gas mixtures,” *International Journal of Heat and Mass Transfer*, vol. 273, Article 129746, 2027, doi: [10.1016/j.ijheatmasstransfer.2026.129746](https://doi.org/10.1016/j.ijheatmasstransfer.2026.129746).

---
## License

This repository is made publicly available under the custom
**PINN-WSGG Academic Research and Non-Commercial License v1.0**. It is not
released under an OSI-approved open-source license.

Academic research, teaching, and other non-commercial use are permitted.
Commercial use, including integration into commercial CFD software,
proprietary simulation platforms, engineering consulting products, or paid
software services, requires prior written permission from the authors.

For commercial licensing, please contact: `tao.ren@sjtu.edu.cn`
