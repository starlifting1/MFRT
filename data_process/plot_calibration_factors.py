#!/usr/bin/env python3
"""Plot the position- and velocity-space classical-limit calibration factors."""

from pathlib import Path
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POSITION_DIAGNOSTIC = REPOSITORY_ROOT / "data/examples_pos/calibration_pos.txt"
VELOCITY_DIAGNOSTIC = REPOSITORY_ROOT / "data/examples_vel/calibration_vel.txt"
DATA_OUTPUT = REPOSITORY_ROOT / "data/examples_pos/calibration_factors_Cr_Cv.pdf"
TEX_OUTPUT = (
    REPOSITORY_ROOT
    / "Substructures_enhanced_Two_Body_Encounters/pics/examples_pos/calibration_factors_Cr_Cv.pdf"
)

FONT_SIZE = 20.0


def isotropic_gaussian_median_factor():
    """Return the analytic median-based C_v for the unsoftened Gaussian limit."""
    lower, upper = 0.0, 4.0
    for _ in range(100):
        y_50 = 0.5 * (lower + upper)
        maxwell_cdf = (
            math.erf(y_50)
            - 2.0 * y_50 / np.sqrt(np.pi) * np.exp(-y_50**2)
        )
        if maxwell_cdf < 0.5:
            lower = y_50
        else:
            upper = y_50
    y_50 = 0.5 * (lower + upper)

    X = np.sqrt(3.0 / 2.0)
    G_X = (
        math.erf(X)
        - 2.0 * X / np.sqrt(np.pi) * np.exp(-X**2)
    ) / (2.0 * X**2)
    classical_coefficient = 3.0 * np.sqrt(6.0 / np.pi) * G_X / X
    raw_median_coefficient = 2.0 * X / y_50 * math.erf(y_50)
    return classical_coefficient / raw_median_coefficient


def main():
    position = np.genfromtxt(POSITION_DIAGNOSTIC, names=True)
    velocity = np.genfromtxt(VELOCITY_DIAGNOSTIC, names=True)

    # In the paper, D^raw uses the corrected estimator formula.  Therefore its
    # C_r is the residual classical-limit factor stored in this diagnostic
    # column, rather than the implementation-only factor for legacy saved files.
    position_N = position["N"]
    C_r = position["C_r_after_GXX"]
    velocity_N = velocity["N"]
    C_v = velocity["C_v"]
    C_v_analytic = isotropic_gaussian_median_factor()

    figure, axes = plt.subplots(1, 2, figsize=(16.0, 6.0))

    axes[0].plot(
        position_N, C_r, "o:", color="tab:blue", linewidth=2.0
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"Particle count $N$", fontsize=FONT_SIZE)
    axes[0].set_ylabel(r"Calibration factor $C_r(N)$", fontsize=FONT_SIZE)
    axes[0].set_title("Position-space factor", fontsize=FONT_SIZE)
    # axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(
        velocity_N, C_v, "o:", color="tab:orange", linewidth=2.0, 
        label="Numerical reference median", 
    )
    axes[1].axhline(
        C_v_analytic, color="black", linestyle="--", linewidth=1.5, 
        label=rf"Isotropic-Gaussian median limit: ${C_v_analytic:.5f}$",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Particle count $N$", fontsize=FONT_SIZE)
    axes[1].set_ylabel(r"Calibration factor $C_v(N)$", fontsize=FONT_SIZE)
    axes[1].set_title("Velocity-space factor", fontsize=FONT_SIZE)
    axes[1].set_ylim(0.34, 0.35)
    # axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(frameon=False, fontsize=FONT_SIZE)

    for axis in axes:
        axis.tick_params(direction="in", which="both", labelsize=FONT_SIZE)

    figure.tight_layout()
    DATA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(DATA_OUTPUT, bbox_inches="tight")
    plt.close(figure)

    TEX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DATA_OUTPUT, TEX_OUTPUT)
    print("wrote", DATA_OUTPUT)
    print("wrote", TEX_OUTPUT)
    print("analytic isotropic-Gaussian median C_v =", C_v_analytic)


if __name__ == "__main__":
    main()
