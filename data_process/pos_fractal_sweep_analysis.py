#!/usr/bin/env python3

import sys
import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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



def plot_ERDC_vs_dimension(records, output_path):

    fig, ax = plt.subplots(figsize=(10, 8))
    fontsize = 24
    pointsize = 20
    
    perlin = [record for record in records if record["sample_type"] == "perlin"]
    reference = [record for record in records if record["sample_type"] == "reference"]

    if perlin:
        iterations = np.array([record["N_iter"] for record in perlin], dtype=float)
        scatter = ax.scatter(
            [record["Dim_frac"] for record in perlin],
            [record["ERDC_median"] for record in perlin],
            # c=iterations, 
            cmap="viridis", s=pointsize, edgecolors="black", linewidths=0.5,
            label="fractal samples",
        )
        # colorbar = fig.colorbar(scatter, ax=ax)
        # colorbar.set_label("Perlin iterations", fontsize=16)
    else:
        raise ValueError("No samples data.")

    ax.axhline(1.0, color="k", linewidth=pointsize/10, linestyle="--")
    
    ax.set_xlabel(r"Fitted fractal dimension $D_\mathrm{frac}$", fontsize=fontsize)
    ax.set_ylabel(r"Position-space enhancement ratio $\zeta_\mathrm{pos}$", fontsize=fontsize)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=fontsize)
    ax.legend(fontsize=fontsize, loc="best")
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
    plot_ERDC_vs_dimension(records, output_dir / "ERDC_pos_versus_dim_frac.pdf")
    # plot_positions_xy(records, output_dir / "fig_positions_xy_all_samples.pdf")
    print(f"Wrote Step 2 products to: {output_dir}")



if __name__ == "__main__":

    main()
