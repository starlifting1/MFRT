# MFRT

MFRT studies how fractal-like spatial structure and anisotropic or non-Gaussian velocity distributions affect Chandrasekhar-style two-body diffusion and collisional relaxation. The calculations use separable position and velocity distributions and compare the diffusion coefficients with homogeneous-position and isotropic-Gaussian reference models.

The diffusion-coefficient algorithm has been updated, including its angular averaging and normalization.

Run `python3 run.py --help` for compilation, model calculations and analysis; for example, `python3 run.py run --space pos --n 40000 --output runs/position_N40000` calculates a fractal-like model and its homogeneous reference.

## Dependencies

- C++17 compiler with OpenMP support, Eigen3, and GNU Make. The Perlin-noise header is included with the source.
- Python 3.10 or later, with NumPy, SciPy and Matplotlib.

License: GPL-3.0.
