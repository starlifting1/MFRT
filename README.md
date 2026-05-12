#README (infinished)

# Introduction

MFRT is program to compare the the two-body collisional relaxation time of usual DF (uniform DF or simulated DF) and noised DF (Multi-Fraction structured DF, i.e. adding Perlin noise to position space and velocity space) of galaxy.

Author: Jianyu Gu et al.

MFRT/diff_r_sample_each/: path to compute the the two-body collisional diffusion coefficient for each particles of Multi-Fraction structured galaxy.

MFRT/data_process/: path to data process and plot.

# Installation and running

## Installation

Environment: Ubuntu 20.04 or higher

Dependencies: `gcc` `g++` (c++17 or higher) `gsl` `eigen3` `openmpi` `python3`

License: GPL-3.0

Compiling:
```bash
cd MFRT/diff_r_sample_each/
make clean; make all
```

## Running

```bash
mkdir snapshots_like_a_galaxy/ samples_observed/ samples_simulated/ samples_pos/ samples_vel/ examples_pos/ examples_vel/

#(unfinished instructions) Then do in a loop like below. Choose some commands in MFRT/diff_r_sample_each/compare.sh.

#compute two-body collisional diffusion coeffients
cd MFRT/diff_r_sample_each/
bash compare.sh

#data process
cd MFRT/data_process/
python3 diffu_samples_pos.py
python3 diffu_samples_vel.py
python3 plot_diffu_figs.py
```
