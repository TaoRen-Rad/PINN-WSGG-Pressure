# PINN-WSGG-Fluent-UDF

> A Fluent compiled UDF implementation of a pressure-dependent Physics-Informed Neural Network (PINN) based Weighted Sum of Gray Gases (WSGG) model for radiative heat transfer in H<sub>2</sub>O-CO<sub>2</sub>-CO participating-gas mixtures.

---

## Overview

This repository provides a self-contained Fluent compiled UDF for the pressure-dependent PINN-WSGG model. The UDF evaluates gray-gas weights and band absorption coefficients from the local thermochemical state and can be directly hooked into Fluent's non-gray radiation model.

**Key features:**

* Pressure-dependent PINN-WSGG model for H<sub>2</sub>O / CO<sub>2</sub> / CO gas mixtures
* Pressure support range: **0.1-80 bar**
* Fluent compiled UDF implementation
* Neural-network output of 9 gray-gas weights and 8 nonzero absorption coefficients
* Transparent gas coefficient fixed as `k_0 = 0`
* Support for Fluent non-gray gray-band radiation interface

---

## Disclaimer

This Fluent UDF package is an academic extension of our research work and is
released only for reference, academic communication, and technical exchange.

It is not a certified industrial tool, not a commercial software component, and
not a guarantee of solver stability, physical correctness, or engineering
applicability in a user's specific case.

Any person or organization that chooses to compile, modify, integrate, or use
this UDF does so at its own risk. Users are solely responsible for validation,
verification, engineering judgment, and any consequences arising from its use.
The authors do not assume responsibility or liability for direct or indirect
losses, errors, damages, or claims related to this package.

---

## Fluent Version

This package has been verified in **ANSYS Fluent 2022 R1** on the current setup.

For Partially Premixed / PDF cases, the current source reads species mass fractions through:

```c
Pdf_Yi(...)
```

For ordinary species-transport cases, switch:

```c
#define WSGG_USE_PDF_YI 1
```

to:

```c
#define WSGG_USE_PDF_YI 0
```

in `pinn_wsgg_fluent.c`, so the UDF reads species mass fractions through:

```c
C_YI(...)
```

If your Fluent installation requires `sg_pdf.h` for PDF-related declarations, switch:

```c
#define WSGG_INCLUDE_SG_PDF 0
```

to:

```c
#define WSGG_INCLUDE_SG_PDF 1
```

If the UDF already compiles with `pdf_props.h`, this switch can remain `0`.

---

## Repository Structure

```text
PINN-WSGG_UDF/
├── pinn_wsgg_fluent.c        # Fluent UDF source file
├── pinn_wsgg_data.h          # Embedded PINN-WSGG network parameters
├── README.md                 # This document
└── LICENSE                   # License information
```

---

## UDF Functions

The compiled library exposes the following UDF hooks:

| Function                 | Fluent hook                      | Purpose                                            |
| ------------------------ | -------------------------------- | -------------------------------------------------- |
| `preFin`                 | Execute On Loading               | Initializes species indices and operating pressure |
| `refTem`                 | Adjust                           | Evaluates PINN-WSGG and writes results into UDMs   |
| `user_gray_band_abs`     | Gray-band absorption coefficient | Returns gray-band absorption coefficients          |
| `user_gray_band_str`     | Emissivity weighting factor      | Returns gray-gas weights                           |
| `calculation`            | Execute At End, optional         | Optional post-iteration operation                  |
| `print_pinn_wsgg_config` | Execute On Demand                | Prints current UDF configuration                   |

---

## Installation and Usage

### 1. Prepare the Files

Place the following files in the Fluent working directory:

```text
pinn_wsgg_fluent.c
pinn_wsgg_data.h
```

---

### 2. Open the Fluent Case

Open the target Fluent case:

```text
File -> Read -> Case...
```

If a previous solution is available, the corresponding data file can also be read:

```text
File -> Read -> Case & Data...
```

If an existing `.dat` file is read, do not reinitialize the flow field unless a fresh simulation is intended.

---

### 3. Allocate UDMs

In Fluent, allocate at least 19 UDMs:

```text
User-Defined -> Memory...
```

Set:

```text
Number of User-Defined Memory Locations = 19
Number of User-Defined Node Memory Locations = 0
```

---

### 4. Compile the UDF

Open the compiled UDF panel:

```text
User-Defined -> Functions -> Compiled...
```

Add:

```text
Source Files:
    pinn_wsgg_fluent.c

Header Files:
    pinn_wsgg_data.h
```

Then click:

```text
Build
Load
```

---

### 5. Check the UDF Configuration

Run:

```text
User-Defined -> Execute On Demand...
```

Select:

```text
print_pinn_wsgg_config::libudf
```

Confirm that the species indices for CO2, H2O, and CO are found correctly.

---

### 6. Hook the Adjust Function

Open:

```text
User-Defined -> Function Hooks...
```

Set:

```text
Adjust = refTem::libudf
```

The `refTem` function updates the local PINN-WSGG gray-gas weights and absorption coefficients during the Fluent iteration.

---

### 7. Enable the Radiation Model

Open:

```text
Models -> Radiation
```

Enable the required radiation model and set the radiation-solver parameters according to the target case.

If Fluent's non-gray gray-band interface is used, set:

```text
Number of Bands = 9
```

For the initial test, it is recommended to use a low-cost radiation setting. After the UDF coupling is confirmed to be stable, the radiation-solver accuracy can be increased for production simulations.

---

### 8. Hook the Gray-Band Absorption Coefficient

Open the material panel:

```text
Materials -> Mixture -> Edit...
```

For `Absorption Coefficient`, select:

```text
user-defined-gray-band
```

Click `Edit...` and select:

```text
user_gray_band_abs::libudf
```

Then click:

```text
Change/Create
```

---

### 9. Hook the Emissivity Weighting Factor

Open:

```text
User-Defined -> Function Hooks...
```

Set:

```text
Emissivity Weighting Factor = user_gray_band_str::libudf
```

Make sure the Adjust hook is still:

```text
Adjust = refTem::libudf
```

---

### 10. Run the Radiation-Coupled Calculation

Run the Fluent calculation with the radiation model enabled:

```text
Run Calculation -> Calculate
```

For the first coupled test, it is recommended to run only one iteration:

```text
Number of Iterations = 1
```

If the case runs successfully, the PINN-WSGG UDF is coupled with Fluent's non-gray radiation solver.

---

## Recommended Workflow Summary

```text
1. Open the Fluent case
2. Allocate 19 UDMs
3. Compile and load the UDF
4. Run print_pinn_wsgg_config
5. Hook refTem to Adjust
6. Enable the radiation model
7. Set Number of Bands = 9 if the non-gray gray-band interface is used
8. Hook user_gray_band_abs as user-defined-gray-band absorption coefficient
9. Hook user_gray_band_str as emissivity weighting factor
10. Run the radiation-coupled calculation
```

---

## License

This project is made publicly available under the custom
**PINN-WSGG Academic Research and Non-Commercial License v1.0**. It is not
released under an OSI-approved open-source license. Academic and other
non-commercial use is permitted. This Fluent UDF package is also provided only
as an academic extension for reference and technical exchange, and all use is
at the user's own risk. Commercial use requires prior written permission from
the authors. See the [`LICENSE`](LICENSE) file for details.
