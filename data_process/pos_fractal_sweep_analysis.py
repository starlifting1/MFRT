#!/usr/bin/env python3

import sys
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import generate_fractal_sample as gfs



SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RUNS_ROOT = PROJECT_DIR / "data" / "samples_pos" / "pos_fractal_sweep"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "data" / "examples_pos" / "pos_fractal_sweep"

def resolve_run_dir(run_name):
    if run_name == "newest":
        run_dirs = [
            p for p in RUNS_ROOT.iterdir()
            if p.is_dir() and p.name.startswith("run_")
        ]
        if not run_dirs:
            raise FileNotFoundError(f"No run_* directories found in {RUNS_ROOT}")

        return max(run_dirs, key=lambda p: p.name)

    run_dir = RUNS_ROOT / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    return run_dir



def read_summary(summary_path):
    """Read the tab-separated C++ summary file whose header starts with '#'."""
    with summary_path.open(newline="") as handle:
        header_line = handle.readline()
        if not header_line.startswith("#"):
            raise ValueError(f"Expected a '#' header in {summary_path}")
        fieldnames = header_line[1:].strip().split("\t")
        return list(csv.DictReader(handle, fieldnames=fieldnames, delimiter="\t"))

def sample_path_for_row(run_dir, row):
    """Use the local sweep directory, not the C++-relative path in summary.txt."""
    sample_name = Path(row["sample_file"]).name
    sample_path = run_dir / "samples" / sample_name
    if not sample_path.is_file():
        raise FileNotFoundError(f"Missing sample for run {row['run_id']}: {sample_path}")
    return sample_path

def nr_path_for_row(run_dir, row):
    if "nr_file" not in row or not row["nr_file"]:
        raise KeyError(f"Missing required nr_file column for run {row['run_id']}")

    nr_name = Path(row["nr_file"]).name
    nr_path = run_dir / "nr_counts" / nr_name
    if not nr_path.is_file():
        raise FileNotFoundError(f"Missing nr-count file for run {row['run_id']}: {nr_path}")
    return nr_path


def fit_dimension_from_nr_file(nr_file, output_dir, run_id):
    data_nr = np.loadtxt(nr_file, dtype=float)
    if data_nr.ndim != 2 or data_nr.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in nr-count file: {nr_file}")

    radii, inside_counts = data_nr[:, 0], data_nr[:, 1]
    h_frac, dim_frac = gfs.calculate_mean_neareast_count_load(
        radii,
        inside_counts,
        save_path=str(output_dir) + "/",
        suffix=f"run_{int(run_id):03d}",
        is_plot=False,
    )
    return float(h_frac), float(dim_frac)

def write_analysis_table(records, output_path):
    fieldnames = [
        "run_id", "sample_type", "Dim_frac", "h_frac", "Diffu_median", "ERDC_median",
        "Diffu_mean", "ERDC_mean", "noise_scale", "step_size", "N_iter", "octaves",
        "persistence", "lacunarity", "seed", "N", "nr_file", "sample_file",
    ]
    with output_path.open("w", newline="") as handle:
        handle.write("# " + "\t".join(fieldnames) + "\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writerows(records)



def model_endpoint_constrained(D_frac, N, q):
    """Empirical trend constrained to zeta_r(0)=N and zeta_r(3)=1."""
    scaled_dimension = np.clip(np.asarray(D_frac, dtype=float) / 3.0, 0.0, 1.0)
    return N ** ((1.0 - scaled_dimension) ** q)


def plot_ERDC_vs_dimension(
    records, output_path, N, fitted_q, q_lower_uncertainty, q_upper_uncertainty
):

    fig, ax = plt.subplots(figsize=(10, 8))
    fontsize = 24
    pointsize = 20
    
    perlin = [record for record in records if record["sample_type"] == "perlin"]

    if perlin:
        ax.scatter(
            [record["Dim_frac"] for record in perlin],
            [record["ERDC_median"] for record in perlin],
            s=pointsize, edgecolors="black", linewidths=0.5,
            label="fractal-like samples",
        )
    else:
        raise ValueError("No samples data.")

    dimension_grid = np.linspace(0.0, 3.0, 800)
    ax.plot(
        dimension_grid,
        model_endpoint_constrained(dimension_grid, N, fitted_q),
        linewidth=2.5,
        label=(
            rf"Empirical fit "
            rf"($q={fitted_q:.2f}^{{+{q_upper_uncertainty:.2f}}}"
            rf"_{{-{q_lower_uncertainty:.2f}}}$)"
        ),
    )
    ax.scatter(
        [0.0, 3.0], [N, 1.0], marker="x", s=75, linewidths=2,
        label="imposed endpoints",
    )
    ax.axhline(1.0, color="k", linewidth=pointsize/10, linestyle="--")
    
    ax.set_xlabel(r"Fitted fractal dimension $D_\mathrm{frac}$", fontsize=fontsize)
    ax.set_ylabel(r"Position-space enhancement ratio $\zeta_r$", fontsize=fontsize)
    ax.set_yscale("log")
    ax.set_xlim(-0.02, 3.02)
    ax.set_ylim(0.8, 1.5 * N)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fontsize)
    ax.legend(fontsize=17, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_positions_xy(records, output_path):
    
    n_samples = len(records)
    n_cols = min(5, n_samples)
    n_rows = math.ceil(n_samples / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows), squeeze=False)
    fontsize = 20
    pointsize = 10

    positions_by_run = []
    max_abs_xy = 0.0
    for record in records:
        positions = np.loadtxt(record["sample_path"], comments="#", usecols=(1, 2, 3))
        positions_by_run.append(positions)
        max_abs_xy = max(max_abs_xy, float(np.percentile(np.abs(positions[:, :2]), 99.5)))

    for ax, record, positions in zip(axes.flat, records, positions_by_run):
        ax.scatter(
            positions[:, 0], positions[:, 1], s=pointsize, alpha=0.75,
            linewidths=0, rasterized=True,
        )
        ax.set_xlim(-max_abs_xy, max_abs_xy)
        ax.set_ylim(-max_abs_xy, max_abs_xy)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
        ax.set_title(
            "run {run_id}: {sample_type}\nscale={noise_scale:g}, iter={N_iter}, step={step_size}, "
            "$D_{{frac}}$={Dim_frac:.3f}".format(**record),
            fontsize=fontsize/2,
        )
        ax.set_xlabel(r"$x$ (kpc)", fontsize=fontsize/2)
        ax.set_ylabel(r"$y$ (kpc)", fontsize=fontsize/2)
        ax.tick_params(labelsize=fontsize/2)

    for ax in axes.flat[n_samples:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)



def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 pos_fractal_sweep_analysis.py <run_name|newest>")

    run_name = sys.argv[1]
    run_dir = resolve_run_dir(run_name).resolve()
    output_dir = DEFAULT_OUTPUT_ROOT / run_dir.name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_summary(run_dir / "summary.txt")
    if not rows:
        raise ValueError(f"No samples found in {run_dir / 'summary.txt'}")

    records = []
    for row in rows:
        if not np.isclose(float(row["Diffu_median"]), float(row["Diffu_q50"]), rtol=1e-12, atol=1e-12):
            raise ValueError(f"Diffu_median and Diffu_q50 disagree for run {row['run_id']}")
        sample_path = sample_path_for_row(run_dir, row)
        nr_path = nr_path_for_row(run_dir, row)
        h_frac, dim_frac = fit_dimension_from_nr_file(nr_path, output_dir, row["run_id"])
        record = {
            "run_id": int(row["run_id"]),
            "sample_type": row["sample_type"],
            "Dim_frac": dim_frac,
            "h_frac": h_frac,
            "Diffu_median": float(row["Diffu_median"]),
            "ERDC_median": float(row["ERDC_median"]),
            "Diffu_mean": float(row["Diffu_mean"]),
            "ERDC_mean": float(row["ERDC_mean"]),
            "noise_scale": float(row["noise_scale"]),
            "step_size": float(row["step_size"]),
            "N_iter": int(row["N_iter"]),
            "octaves": int(row["octaves"]),
            "persistence": float(row["persistence"]),
            "lacunarity": float(row["lacunarity"]),
            "seed": int(row["seed"]),
            "N": int(row["N"]),
            "nr_file": nr_path.name,
            "sample_file": sample_path.name,
            "sample_path": sample_path,
        }
        records.append(record)
        print(
            f"run {record['run_id']:03d}: D_frac={record['Dim_frac']:.6f}, "
            f"Diffu_median={record['Diffu_median']:.8g}, ERDC_median={record['ERDC_median']:.6f}"
        )

    write_analysis_table(records, output_dir / "pos_fractal_erdc_vs_dfrac.txt")

    # Fit the endpoint-constrained empirical trend to the actual sweep results.
    # The residuals are evaluated in log(zeta_r), as in the exploratory QA fit.
    perlin_records = [record for record in records if record["sample_type"] == "perlin"]
    if not perlin_records:
        raise ValueError("No perlin samples available for the ERDC fit.")

    particle_counts = {record["N"] for record in perlin_records}
    if len(particle_counts) != 1:
        raise ValueError(f"Expected one fixed particle count, found: {sorted(particle_counts)}")
    particle_count = float(particle_counts.pop())

    dimensions = np.array([record["Dim_frac"] for record in perlin_records], dtype=float)
    enhancements = np.array([record["ERDC_median"] for record in perlin_records], dtype=float)
    if np.any((dimensions < 0.0) | (dimensions > 3.0)):
        raise ValueError("The endpoint-constrained fit requires 0 <= D_frac <= 3.")
    if np.any(enhancements <= 0.0):
        raise ValueError("The log-space fit requires positive enhancement ratios.")

    fit_result = least_squares(
        lambda parameters: (
            np.log(model_endpoint_constrained(dimensions, particle_count, parameters[0]))
            - np.log(enhancements)
        ),
        x0=[4.0],
        bounds=([0.01], [50.0]),
        loss="soft_l1",
        f_scale=0.3,
    )
    if not fit_result.success:
        raise RuntimeError(f"ERDC fit failed: {fit_result.message}")

    fitted_q = float(fit_result.x[0])
    residual_dex = (
        np.log10(model_endpoint_constrained(dimensions, particle_count, fitted_q))
        - np.log10(enhancements)
    )
    rmse_dex = float(np.sqrt(np.mean(residual_dex**2)))

    # Point-wise non-parametric bootstrap uncertainty for q. The fixed seed makes
    # the reported central 68% interval reproducible.
    bootstrap_rng = np.random.default_rng(20260906)
    bootstrap_q = np.empty(500, dtype=float)
    for bootstrap_index in range(bootstrap_q.size):
        sample_indices = bootstrap_rng.integers(
            0, len(perlin_records), size=len(perlin_records)
        )
        bootstrap_dimensions = dimensions[sample_indices]
        bootstrap_enhancements = enhancements[sample_indices]
        bootstrap_result = least_squares(
            lambda parameters: (
                np.log(
                    model_endpoint_constrained(
                        bootstrap_dimensions, particle_count, parameters[0]
                    )
                )
                - np.log(bootstrap_enhancements)
            ),
            x0=[fitted_q],
            bounds=([0.01], [50.0]),
            loss="soft_l1",
            f_scale=0.3,
        )
        if not bootstrap_result.success:
            raise RuntimeError(
                f"Bootstrap ERDC fit {bootstrap_index} failed: "
                f"{bootstrap_result.message}"
            )
        bootstrap_q[bootstrap_index] = bootstrap_result.x[0]

    q_lower, q_upper = np.percentile(bootstrap_q, [16.0, 84.0])
    q_lower_uncertainty = float(fitted_q - q_lower)
    q_upper_uncertainty = float(q_upper - fitted_q)
    print(
        f"Endpoint-constrained fit: q={fitted_q:.8f} "
        f"(-{q_lower_uncertainty:.8f}, +{q_upper_uncertainty:.8f}; "
        f"central 68% bootstrap interval), log-space RMSE={rmse_dex:.6f} dex"
    )

    figure_path = output_dir / "ERDC_pos_versus_dim_frac.pdf"
    plot_ERDC_vs_dimension(
        records,
        figure_path,
        particle_count,
        fitted_q,
        q_lower_uncertainty,
        q_upper_uncertainty,
    )
    # Also write the stable path used when collecting paper-facing figures.
    plot_ERDC_vs_dimension(
        records,
        DEFAULT_OUTPUT_ROOT / "ERDC_pos_versus_dim_frac.pdf",
        particle_count,
        fitted_q,
        q_lower_uncertainty,
        q_upper_uncertainty,
    )
    # plot_positions_xy(records, output_dir / "fig_positions_xy_all_samples.pdf")
    print(f"Wrote Step 2 products to: {output_dir}")



if __name__ == "__main__":

    main()
