# README

## Introduction

MFRT is a numerical code for quantifying, within separable phase-space distribution functions (DFs), how fractal-like position-space substructure and anisotropic or non-Gaussian velocity-space substructure modify Chandrasekhar-style two-body diffusion and collisional relaxation relative to homogeneous-position and isotropic-Gaussian reference distributions.

Author: Jianyu Gu et al.

`MFRT/diffu_r_simple_each/`: C++ code for calculating two-body collisional diffusion coefficients.

`MFRT/data_process/`: path to data process and plot.

## Installation

Environment: Ubuntu 20.04 or higher

Dependencies: `GNU Make`; `G++` (C++17 with OpenMP support), `Eigen3`; `Python 3`, `Astropy`, `Matplotlib`, `noise`, `NumPy`, `pandas`, `scikit-learn`, `SciPy`, `tikzplotlib`, and `vegas`.

License: GPL-3.0

Compiling:
```bash
cd MFRT/diffu_r_simple_each/
make clean; make all
```

## Running

### calculation

- `main.cpp` (compiled as `out.exe`) generates or reads position-space samples and calculates the position-space diffusion coefficient for each particle.
- `main_vel.cpp` (compiled as `out_vel.exe`) generates isotropic-Gaussian, anisotropic, power-law-tailed, fractal-like, and composite velocity-space samples and calculates their diffusion tensors.
- `main_pos_fractal.cpp` (compiled as `main_pos_fractal.exe`) performs the position-space fractal-parameter sweep and records the diffusion enhancement relative to the homogeneous reference sample.

### plot

```
cd MFRT/data_process/
python3 diffu_samples_pos.py
python3 diffu_samples_vel.py
python3 plot_diffu_figs.py
```
