# PINN-WSGG-OpenFOAM

> An OpenFOAM implementation of a pressure-dependent Physics-Informed Neural Network (PINN) based Weighted Sum of Gray Gases (WSGG) model for radiative heat transfer in H<sub>2</sub>O-CO<sub>2</sub>-CO participating-gas mixtures.

---

## Overview

This repository provides the OpenFOAM source code for a pressure-dependent PINN-WSGG radiation model together with a `thermoFoam` demonstration case. The package contains the released `nonGreyP1` radiation implementation, the deployed `wsgg.json` neural-network parameters, and local build scripts for compiling the required shared libraries.

**Key features:**

- Pressure-dependent PINN-WSGG model for H<sub>2</sub>O / CO<sub>2</sub> / CO gas mixtures
- OpenFOAM `nonGreyP1` radiation model implementation
- Neural-network output of 9 gray-gas weights and 8 nonzero absorption coefficients
- Transparent gas coefficient fixed as `k_0 = 0`
- Bundled `ModestSandiaD` example case for `thermoFoam`

---

## OpenFOAM Version

This package is set up for **OpenFOAM v2412** on the current environment.

Before building or running, load the OpenFOAM environment:

```bash
source /opt/OpenFOAM/OpenFOAM-v2412/etc/bashrc
```

If your OpenFOAM installation is in a different location, replace the path accordingly.

---

## Build

Enter the package directory:

```bash
cd ./PINN-WSGG_OpenFOAM
```

Build the local libraries:

```bash
export PINN_WSGG_ROOT=$(pwd)
./src/Allwmake
```

Set the local library path before running the case:

```bash
export FOAM_USER_LIBBIN=$PINN_WSGG_ROOT/platforms/$WM_OPTIONS/lib
export LD_LIBRARY_PATH=$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH
```

To remove locally built files:

```bash
./src/Allwclean
```

---

## Run the Example Case

Run the bundled `thermoFoam` case:

```bash
thermoFoam -case ./ModestSandiaD
```

The current released case does **not** require the extra function objects that existed in earlier development snapshots. The example now runs with only:

- `libglobalModelParameter.so`
- `libglobalModels.so`

---

## Repository Structure

```text
PINN-WSGG_OpenFOAM/
├── ModestSandiaD/                    # Example thermoFoam case
│   ├── 0/                            # Initial fields
│   ├── 1/                            # Example output fields
│   ├── chemkin/                      # Chemistry and thermo input files
│   ├── constant/
│   ├── postProcessing/
│   └── system/
├── platforms/                        # Local compiled libraries
├── src/                              # OpenFOAM source code
│   ├── Allwmake
│   ├── Allwclean
│   ├── globalModelParameter/
│   │   ├── globalModelParameter/     # Core parameter library
│   │   └── parameter/
│   │       └── wsgg.json             # Deployed PINN-WSGG model parameters
│   └── radiation/                    # Radiation library source
├── README.md                         # This document
├── LICENSE                           # License information
├── requirements.txt                  # Environment requirements
└── .gitignore
```

---

## Notes

- The current package is a reduced release focused on the PINN-WSGG + `nonGreyP1` workflow.
- The previously used auxiliary function objects for volume-fraction and Planck-mean post-processing are not part of the current runnable path.
- The example case has been verified locally with `thermoFoam -case ./ModestSandiaD` after compiling the libraries in this package.

---

## License

This project is made publicly available under the custom
**PINN-WSGG Academic Research and Non-Commercial License v1.0**. It is not
released under an OSI-approved open-source license. Academic and other
non-commercial use is permitted. Commercial use requires prior written
permission from the authors. See the [`LICENSE`](LICENSE) file for details.
