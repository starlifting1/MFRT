"""Statistics and plots for mfrt_trace_v2 diffusion outputs.

Public API: analyze_root(root, output, make_plots=True).
All diffusion statistics use Dtot = Tr(D). Calibration uses a single factor
from the matched numerical reference median for every particle and statistic.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE = {"pos": "uniform", "vel": "iso"}
# Structure parameters intentionally do not enter reference matching.
MATCH_FIELDS = ("R0", "Rm", "v0", "M_total", "mass", "G", "sphere_radius", "epsilon", "v_epsilon",
                "eps_factor", "veps_factor", "theta", "crossing_length")
QUANTILES = ("q16", "median", "q84", "q95", "q99")
DIFFUSION_LABEL = r"$D_{\rm tot}$ [$(\mathrm{km}\,\mathrm{s}^{-1})^3\,\mathrm{kpc}^{-1}$]"
PLOT_PARAMETERS = ("R0", "Rm", "v0", "mass", "theta", "eps_factor", "veps_factor",
                   "axes", "noise_scale", "noise_step", "iterations", "octaves", "persistence", "lacunarity")
COMBINED_PHYSICAL_FIELDS = ("R0", "Rm", "v0", "mass", "G", "sphere_radius", "epsilon",
                            "eps_factor", "ln_Lambda", "crossing_length", "t_cross",
                            "velocity_second_moment")
REALIZATION_FIELDS = tuple(dict.fromkeys(
    ("schema", "space", "kind", "n") + MATCH_FIELDS + PLOT_PARAMETERS
    + ("perlin_seed", "ln_Lambda", "t_cross", "velocity_second_moment")))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def read_table(path, min_columns, allow_empty=False):
    """Accept both # headers and one un-commented column-name header."""
    path = Path(path)
    skip = 0
    with path.open() as handle:
        for number, line in enumerate(handle):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                [float(word) for word in line.split()]
            except ValueError:
                skip = number + 1
                continue
            break
        else:
            if allow_empty:
                return np.empty((0, min_columns))
            raise ValueError(f"Empty table: {path}")
    array = np.loadtxt(path, comments="#", skiprows=skip, ndmin=2)
    if array.shape[1] < min_columns or not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite values or insufficient columns: {path}")
    return array


def particle_table(path, min_columns, allow_empty=False):
    array = read_table(path, min_columns, allow_empty=allow_empty)
    ids = array[:, 0]
    if np.any(ids < 0) or np.any(ids != np.floor(ids)) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"Particle IDs must be unique nonnegative integers: {path}")
    return array


def align_ids(source, target_ids):
    order = np.argsort(source[:, 0])
    ids = source[order, 0]
    indices = np.searchsorted(ids, target_ids)
    if np.any(indices >= len(ids)) or np.any(ids[np.minimum(indices, len(ids)-1)] != target_ids):
        raise ValueError("Target particle ID absent from source table")
    return source[order[indices]]


def statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Statistics require a nonempty finite array")
    q = np.quantile(values, [.16, .5, .84, .95, .99])
    return dict(n=len(values), mean=float(np.mean(values)), min=float(np.min(values)),
                max=float(np.max(values)), **dict(zip(QUANTILES, map(float, q))))


def error_quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return dict(n=len(values), p50=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=float(np.max(values)))


def relative_error(tree, direct):
    valid = direct != 0
    return (tree[valid] - direct[valid]) / direct[valid]


def tensor_diagnostics(table):
    if table.shape[1] < 8:
        return None
    minimum, negative, mismatch = np.inf, 0, 0.
    for start in range(0, len(table), 100000):
        rows = table[start:start+100000]
        tensor = np.empty((len(rows), 3, 3))
        tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2] = rows[:, 2], rows[:, 3], rows[:, 4]
        tensor[:, 0, 1] = tensor[:, 1, 0] = rows[:, 5]
        tensor[:, 0, 2] = tensor[:, 2, 0] = rows[:, 6]
        tensor[:, 1, 2] = tensor[:, 2, 1] = rows[:, 7]
        eig = np.linalg.eigvalsh(tensor)[:, 0]
        scale = np.maximum(np.abs(rows[:, 1]), np.finfo(float).tiny)
        minimum = min(minimum, float(eig.min()))
        negative += int(np.count_nonzero(eig < -1.e-10 * scale))
        mismatch = max(mismatch, float(np.max(np.abs(rows[:, 1] - rows[:, 2:5].sum(axis=1)) / scale)))
    return {"min_eigenvalue": minimum, "negative_eigenvalue_fraction": negative / len(table),
            "negative_eigenvalue_relative_tolerance": 1.e-10,
            "max_trace_relative_mismatch": mismatch}


def tree_errors(tree, direct):
    selected = align_ids(tree, direct[:, 0])
    relative = relative_error(selected[:, 1], direct[:, 1])
    result = {
        "n_targets": len(direct),
        "full_population_direct": len(direct) == len(tree),
        "target_selection_note": ("All source particles" if len(direct) == len(tree)
                                  else "Target subset can include deliberately selected extreme diffusion and outermost particles; its mean is not an unbiased population estimate"),
        "trace_absolute_error": error_quantiles(np.abs(selected[:, 1] - direct[:, 1])),
        "trace_absolute_relative_error": error_quantiles(np.abs(relative)),
        "trace_signed_relative_median": float(np.median(relative)) if len(relative) else None,
        "zero_direct_trace_count": int(np.count_nonzero(direct[:, 1] == 0)),
        "tree_target_statistics": statistics(selected[:, 1]),
        "direct_target_statistics": statistics(direct[:, 1]),
        "tree_tensor": tensor_diagnostics(selected), "direct_tensor": tensor_diagnostics(direct),
    }
    for statistic in ("median", "mean"):
        td = result["direct_target_statistics"][statistic]
        tt = result["tree_target_statistics"][statistic]
        result[statistic + "_relative_error"] = tt / td - 1 if td else None
    if selected.shape[1] >= 8 and direct.shape[1] >= 8:
        components = {}
        for col, name in enumerate(("D11", "D22", "D33", "D12", "D13", "D23"), 2):
            delta = np.abs(selected[:, col] - direct[:, col])
            item = {"absolute_error": error_quantiles(delta)}
            if col < 5:
                item["absolute_relative_error"] = error_quantiles(np.abs(relative_error(selected[:, col], direct[:, col])))
            else:
                valid = direct[:, 1] != 0
                item["absolute_error_over_direct_trace"] = error_quantiles(delta[valid] / np.abs(direct[valid, 1]))
            components[name] = item
        result["components"] = components
    return result


def fit_counts(radii, counts, fit_r_min=None, fit_r_max=None):
    """Historical mask: all positive saved cumulative counts unless bounded.

    OLS log N(<r) = log h + D log r equals the historical fit to
    log sphere-averaged density. No radius interval or dimension anchor is added.
    """
    radii, counts = np.asarray(radii), np.asarray(counts)
    if any(bound is not None and (not np.isfinite(bound) or bound <= 0) for bound in (fit_r_min, fit_r_max)):
        raise ValueError("Fit bounds must be positive finite distances")
    if fit_r_min is not None and fit_r_max is not None and fit_r_min > fit_r_max:
        raise ValueError("Fit lower bound exceeds upper bound")
    if (len(radii) < 2 or np.any(radii <= 0) or np.any(counts < 0)
            or not np.all(np.isfinite(radii)) or not np.all(np.isfinite(counts))
            or np.any(np.diff(radii) <= 0) or np.any(np.diff(counts) < -1.e-10 * max(1., counts.max()))):
        raise ValueError("Invalid cumulative neighbour-count table")
    mask = counts > 0
    if fit_r_min is not None:
        mask &= radii >= fit_r_min
    if fit_r_max is not None:
        mask &= radii <= fit_r_max
    local = np.full(len(radii)-1, np.nan)
    pairs = (counts[:-1] > 0) & (counts[1:] > 0)
    local[pairs] = np.log(counts[1:][pairs] / counts[:-1][pairs]) / np.log(radii[1:][pairs] / radii[:-1][pairs])
    result = {"count_r_min": float(radii[0]), "count_r_max": float(radii[-1]),
              "local_radius": np.sqrt(radii[:-1] * radii[1:]), "local_D": local,
              "fit_mask": mask, "n_fit_radii": int(mask.sum()),
              "statistic": "mean cumulative neighbour count excluding source particle by index"}
    if mask.sum() < 2:
        return {**result, "D_frac": None, "fit_issue": "Fewer than two positive bins in requested interval"}
    dimension, log_h = np.polyfit(np.log10(radii[mask]), np.log10(counts[mask]), 1)
    residual = np.log10(counts[mask]) - log_h - dimension*np.log10(radii[mask])
    h = float(10**log_h)
    return {**result, "D_frac": float(dimension), "h_count": h,
            "h_volume_mean_density": 3*h/(4*np.pi), "h_shell_density": dimension*h/(4*np.pi),
            "h_length_exponent": -float(dimension),
            "fit_r_min": float(radii[mask][0]), "fit_r_max": float(radii[mask][-1]),
            "log10_count_residual_rms": float(np.sqrt(np.mean(residual**2))),
            "uncertainty_note": "Fit residuals do not measure realization uncertainty; bins share pairs"}


def _equivalent(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(a, b, rtol=1.e-10, atol=0))
    return a == b


def match_references(case, cases):
    metadata = case["metadata"]
    candidates = []
    for reference in cases:
        ref = reference["metadata"]
        if ref["kind"] != REFERENCE[metadata["space"]] or ref["space"] != metadata["space"] or ref["n"] != metadata["n"]:
            continue
        if all(_equivalent(metadata.get(key), ref.get(key)) for key in MATCH_FIELDS):
            candidates.append(reference)
    same_seed = [ref for ref in candidates if ref["metadata"]["seed"] == metadata["seed"]]
    if case["metadata"]["kind"] == REFERENCE[metadata["space"]]:
        return [case], "self_reference"
    return (same_seed, "same_seed") if same_seed else (candidates, "other_seed_reference_ensemble")


def correlation(x, y):
    if len(x) < 2:
        return None
    dx, dy = x - np.mean(x), y - np.mean(y)
    denominator = np.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
    return float(np.dot(dx, dy) / denominator) if denominator else None


def norm_correlations(norm, diffusion):
    positive = (norm > 0) & (diffusion > 0)
    result = {"pearson_D_norm": correlation(norm, diffusion),
              "pearson_logD_logNorm": correlation(np.log(norm[positive]), np.log(diffusion[positive]))}
    # Rank correlation on a deterministic bounded display sample; full Pearson above.
    from scipy.stats import rankdata
    indices = np.linspace(0, len(norm)-1, min(100000, len(norm)), dtype=int)
    result["spearman_D_norm"] = correlation(rankdata(norm[indices]), rankdata(diffusion[indices]))
    result["spearman_sample_size"] = len(indices)
    return result


def nearest_velocity_diagnostics(coordinates, v0, diffusion=None):
    """Finite-sample velocity separations; not a strong-deflection model test."""
    from scipy.spatial import cKDTree
    if v0 <= 0:
        raise ValueError("v0 must be positive for nearest-velocity diagnostics")
    result = {"n_particles": len(coordinates), "normalization_v0": float(v0),
              "method": "scipy.spatial.cKDTree k=2; exclude one self entry",
              "interpretation": "Finite-sample separation diagnostic only. Absence of exact duplicate velocities does not validate strong-deflection physics or a complete scattering model."}
    if len(coordinates) < 2:
        return {**result, "min_nearest_u_over_v0": None, "quantile_probabilities": [.001, .01, .5],
                "nearest_u_over_v0_quantiles": [None, None, None], "exact_zero_nearest_count": 0,
                "exact_duplicate_particle_count": 0, "exact_zero_pair_count": 0}
    distances, _ = cKDTree(coordinates).query(coordinates, k=2, workers=1)
    nearest = distances[:, 1]
    zero_mask = nearest == 0
    # Restrict the exact-coordinate check to the zero-distance candidates.
    # This also distinguishes exact duplicates from distance underflow.
    duplicate_counts = np.unique(coordinates[zero_mask], axis=0, return_counts=True)[1]
    duplicate_counts = duplicate_counts[duplicate_counts > 1]
    result.update({"min_nearest_u_over_v0": float(np.min(nearest)/v0),
            "quantile_probabilities": [.001, .01, .5],
            "nearest_u_over_v0_quantiles": np.quantile(nearest/v0, [.001, .01, .5]),
            "exact_zero_nearest_count": int(np.count_nonzero(zero_mask)),
            "exact_duplicate_particle_count": int(duplicate_counts.sum()),
            "exact_zero_pair_count": int(np.sum(duplicate_counts*(duplicate_counts-1)//2))})
    if diffusion is not None:
        concentrations = []
        total = np.sum(diffusion)
        for probability, threshold in zip((.001, .01, .5), np.quantile(nearest, [.001, .01, .5])):
            selected = nearest <= threshold
            concentrations.append({"nearest_quantile": probability,
                                   "actual_particle_fraction": float(np.mean(selected)),
                                   "Dtot_sum_fraction": float(np.sum(diffusion[selected])/total) if total > 0 else None,
                                   "Dtot_mean": float(np.mean(diffusion[selected]))})
        result["diffusion_in_small_nearest_distance_subset"] = concentrations
    return result


def time_diagnostics(diffusion, metadata):
    second_moment = metadata.get("velocity_second_moment", metadata["v0"]**2)
    crossing_length = metadata.get("crossing_length", metadata["R0"])
    crossing_time = metadata.get("t_cross", crossing_length / metadata["v0"])
    if crossing_time <= 0 or second_moment <= 0:
        raise ValueError("Crossing time and velocity second moment must be positive")
    ratios = np.full(len(diffusion), np.inf)
    np.divide(second_moment, crossing_time * diffusion, out=ratios, where=diffusion > 0)
    finite = ratios[np.isfinite(ratios)]
    return {"definition": "tdiff=<v^2>/Dtot; tcross=crossing_length/v0 (default R0/v0)",
            "crossing_length": crossing_length, "t_cross": crossing_time,
            "velocity_second_moment": second_moment,
            "finite_ratio_statistics": statistics(finite) if len(finite) else None,
            "infinite_ratio_fraction": float(np.mean(~np.isfinite(ratios))),
            "fraction_below": {str(limit): float(np.mean(ratios < limit)) for limit in (10, 100, 1000)}}


def _scaled_statistics(raw, factor):
    return {key: value if key == "n" else value * factor for key, value in raw.items()}


def _reference_stat(refs, key):
    # Equal weight per independent realization; not a pooled-particle median.
    values = [ref["summary"]["raw"][key] for ref in refs]
    return float(np.median(values)) if key == "median" else float(np.mean(values))


def _enhancement_errors(error, refs, space):
    available = [ref["tree_error"] for ref in refs if ref.get("tree_error")]
    if not available:
        return None
    full_population = error.get("full_population_direct", False) and all(r.get("full_population_direct", False) for r in available)
    result = {}
    for stat in ("median", "mean"):
        aggregate = np.median if stat == "median" else np.mean
        direct_ref = aggregate([r["direct_target_statistics"][stat] for r in available])
        tree_ref = aggregate([r["tree_target_statistics"][stat] for r in available])
        direct_mod = error["direct_target_statistics"][stat]
        tree_mod = error["tree_target_statistics"][stat]
        if direct_ref <= 0 or tree_ref <= 0:
            continue
        direct_zeta, tree_zeta = float(direct_mod/direct_ref), float(tree_mod/tree_ref)
        result[stat] = {"direct_zeta": direct_zeta, "tree_zeta": tree_zeta,
                        "absolute_zeta_error": abs(tree_zeta-direct_zeta),
                        "relative_zeta_error": tree_zeta/direct_zeta-1 if direct_zeta else None}
        if space == "vel":
            tolerance = min(.005, abs(direct_zeta-1)/3)
            result[stat].update(absolute_zeta_tolerance=tolerance,
                                passes_effect_resolution=abs(tree_zeta-direct_zeta) <= tolerance)
        else:
            result[stat].update(relative_zeta_tolerance=.03,
                                passes_effect_resolution=abs(tree_zeta/direct_zeta-1) <= .03 if direct_zeta else False)
        result[stat]["strict_flag"] = result[stat]["passes_effect_resolution"]
        result[stat]["used_for_full_population_effect_acceptance"] = bool(full_population)
        if not full_population:
            result[stat]["passes_effect_resolution"] = None
            result[stat]["acceptance_note"] = "Strict target-subset flag retained as a diagnostic only. The subset includes selected extremes and does not estimate the full-population median or mean enhancement. Use full-population checks and per-target errors."
    result["comparison_scope"] = "identical tree/direct target IDs within each case; reference target sample may differ"
    return result


def _load_diffusion(case):
    table = particle_table(case["directory"] / "diffusion.tsv", 8 if case["metadata"]["space"] == "vel" else 2)
    if len(table) != int(case["metadata"]["n"]) or np.any(table[:, 1] < 0):
        raise ValueError("Diffusion count differs from n, or Dtot is negative")
    return table


def _case_analysis(case, refs, mode):
    metadata, directory = case["metadata"], case["directory"]
    diffusion = _load_diffusion(case)
    raw = statistics(diffusion[:, 1])
    # A reference can analyze itself before its summary has been stored.
    reference_median = raw["median"] if mode == "self_reference" else _reference_stat(refs, "median") if refs else None
    reference_mean = raw["mean"] if mode == "self_reference" else _reference_stat(refs, "mean") if refs else None
    classical_median = metadata["D_classical_median"]
    if classical_median <= 0 or raw["median"] <= 0:
        raise ValueError("Classical and numerical median diffusion must be positive")
    factor = classical_median/reference_median if reference_median else None
    calibrated = diffusion[:, 1] * factor if factor else None
    mean_classical = (classical_median if metadata["space"] == "pos"
                      else metadata.get("D_classical_mean", classical_median))
    summary = {"case_id": case["case_id"], "directory": str(directory), "metadata": metadata,
               "raw": raw, "calibrated": _scaled_statistics(raw, factor) if factor else None,
               "reference_cases": [ref["case_id"] for ref in refs], "reference_match": mode if refs else "missing",
               "reference_aggregation": "median of realization medians; mean of realization means",
               "calibration_factor": factor,
               "mean_based_calibration_factor_diagnostic": mean_classical/reference_mean if reference_mean else None,
               "mean_calibration_classical_diagnostic": mean_classical,
               "mean_calibration_convention": ("Position mean at fixed Maxwellian v50; classical coefficient remains D_classical_median"
                                               if metadata["space"] == "pos" else
                                               "Mean over target velocities; use D_classical_mean"),
               "zeta_median": raw["median"]/reference_median if reference_median else None,
               "zeta_mean": raw["mean"]/reference_mean if reference_mean else None,
               "raw_fraction_above_3_classical_median": float(np.mean(diffusion[:, 1] > 3*classical_median)),
               "calibrated_fraction_above_3_classical_median": float(np.mean(calibrated > 3*classical_median)) if factor else None,
               "raw_time": time_diagnostics(diffusion[:, 1], metadata),
               "calibrated_time": time_diagnostics(calibrated, metadata) if factor else None,
               "tensor": tensor_diagnostics(diffusion)}
    samples = particle_table(directory / "samples.tsv", 4)
    if len(samples) != len(diffusion):
        raise ValueError("Sample and diffusion lengths differ")
    coordinates = align_ids(samples, diffusion[:, 0])[:, 1:4]
    norm = np.linalg.norm(coordinates, axis=1)
    summary["coordinate_norm_name"] = "radius" if metadata["space"] == "pos" else "speed"
    summary["coordinate_norm_statistics"] = statistics(norm)
    summary["coordinate_norm_rms"] = float(np.sqrt(np.mean(norm**2)))
    summary["coordinate_mean"] = np.mean(coordinates, axis=0)
    summary["diffusion_norm_correlations"] = norm_correlations(norm, diffusion[:, 1])
    if metadata["space"] == "vel":
        summary["nearest_velocity"] = nearest_velocity_diagnostics(coordinates, metadata["v0"], diffusion[:, 1])
    if (directory / "nr.tsv").exists():
        nr = read_table(directory / "nr.tsv", 2)
        summary["fractal"] = fit_counts(nr[:, 0], nr[:, 1], metadata.get("fit_r_min"), metadata.get("fit_r_max"))
        scale = metadata["Rm"] if metadata["space"] == "pos" else metadata["v0"]
        summary["fractal"].update(scale_value=scale, scale_name="Rm" if metadata["space"] == "pos" else "v0")
        for key in ("count_r_min", "count_r_max", "fit_r_min", "fit_r_max"):
            if key in summary["fractal"]:
                summary["fractal"][key+"_over_scale"] = summary["fractal"][key]/scale
    else:
        nr = None
        summary["fractal"] = None
    diagnostic_path = directory / "nr_diagnostic.tsv"
    nr_diagnostic = read_table(diagnostic_path, 2) if diagnostic_path.exists() else None
    if nr_diagnostic is not None:
        if np.any(nr_diagnostic[:, 0] <= 0) or np.any(nr_diagnostic[:, 1] < 0):
            raise ValueError("Invalid diagnostic neighbour counts")
        summary["diagnostic_count_coverage"] = {
            "count_r_min": float(nr_diagnostic[:, 0].min()),
            "count_r_max": float(nr_diagnostic[:, 0].max()),
            "n_radii": len(nr_diagnostic), "used_for_main_fit": False}
    direct = None
    if (directory / "direct.tsv").exists():
        direct = particle_table(directory / "direct.tsv", 8 if metadata["space"] == "vel" else 2, allow_empty=True)
    if "direct_targets" in metadata and metadata["direct_targets"] != (len(direct) if direct is not None else 0):
        raise ValueError(f"direct_targets metadata differs from actual direct rows: {directory}")
    error = tree_errors(diffusion, direct) if direct is not None and len(direct) else None
    summary["direct_status"] = ("disabled_by_metadata" if metadata.get("direct_targets") == 0
                                else "available" if error else "empty_or_absent")
    if error:
        error["enhancement_errors"] = _enhancement_errors(error, refs, metadata["space"]) if mode != "self_reference" else None
    indices = np.linspace(0, len(diffusion)-1, min(50000, len(diffusion)), dtype=int)
    display = {"norm": norm[indices], "coordinates": coordinates[indices], "Dtot": diffusion[indices, 1],
               # Copy one column: retaining a view would keep the whole tensor table alive.
               "distribution_values": diffusion[:, 1].copy(),
               "nr": nr, "nr_diagnostic": nr_diagnostic}
    return summary, error, display


def _plot_case(case, output):
    import matplotlib.pyplot as plt
    summary, display = case["summary"], case["display"]
    stem = hashlib.sha256(case["case_id"].encode()).hexdigest()[:10] + "_" + case["metadata"]["kind"]
    fig, ax = plt.subplots(figsize=(5, 4))
    positive = (display["norm"] > 0) & (display["Dtot"] > 0)
    ax.scatter(display["norm"][positive], display["Dtot"][positive], s=2, alpha=.2, rasterized=True)
    unit = "kpc" if case["metadata"]["space"] == "pos" else r"km s$^{-1}$"
    ax.set(xscale="log", yscale="log", xlabel=summary["coordinate_norm_name"] + " [" + unit + "]", ylabel=DIFFUSION_LABEL,
           title=f"{case['metadata']['kind']}, N={case['metadata']['n']}, seed={case['metadata']['seed']}\n"
                 f"R0={case['metadata']['R0']} kpc, theta={case['metadata'].get('theta')}, veps-factor={case['metadata'].get('veps_factor', 1)}")
    fig.tight_layout(); fig.savefig(output / (stem + "_scatter.pdf")); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    coordinates = display["coordinates"]
    for ax, second, label in ((axes[0], 1, "y"), (axes[1], 2, "z")):
        ax.scatter(coordinates[:, 0], coordinates[:, second], s=2, alpha=.2, rasterized=True)
        first_name, second_name = (r"$v_x$", "$v_"+label+"$") if case["metadata"]["space"] == "vel" else ("x", label)
        ax.set(xlabel=first_name+" ["+unit+"]", ylabel=second_name+" ["+unit+"]")
        ax.set_aspect("equal", adjustable="box")
    axes[2].hist(display["norm"], bins=50, density=True, histtype="step")
    axes[2].set(xlabel=summary["coordinate_norm_name"]+" ["+unit+"]", ylabel="probability density")
    if case["metadata"]["space"] == "vel":
        axes[2].set_yscale("log")
    fig.suptitle(case["case_id"] + "; display <=50000 particles", fontsize=9)
    fig.tight_layout(); fig.savefig(output / (stem+"_structure.pdf")); plt.close(fig)
    if display["nr"] is not None:
        r, counts = display["nr"].T[:2]
        fit = summary["fractal"]
        fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
        if display.get("nr_diagnostic") is not None:
            diagnostic = display["nr_diagnostic"]
            axes[0].loglog(diagnostic[:, 0], diagnostic[:, 1], ".-", color="grey", alpha=.35,
                          label="diagnostic coverage (not used for fit)")
            positive = (diagnostic[:-1, 1] > 0) & (diagnostic[1:, 1] > 0)
            midpoints = np.sqrt(diagnostic[:-1, 0] * diagnostic[1:, 0])
            slopes = np.full(len(midpoints), np.nan)
            slopes[positive] = np.log(diagnostic[1:, 1][positive]/diagnostic[:-1, 1][positive])/np.log(
                diagnostic[1:, 0][positive]/diagnostic[:-1, 0][positive])
            axes[1].semilogx(midpoints, slopes, ".-", color="grey", alpha=.35)
        axes[0].loglog(r, counts, ".-", label="saved cumulative count")
        if fit.get("D_frac") is not None:
            mask = fit["fit_mask"]
            axes[0].loglog(r[mask], fit["h_count"]*r[mask]**fit["D_frac"], "--", label=f"D={fit['D_frac']:.3f}")
            for ax in axes:
                ax.axvspan(fit["fit_r_min"], fit["fit_r_max"], color="C1", alpha=.12)
        axes[0].set(ylabel=r"$\langle N(<r)\rangle$" if case["metadata"]["space"] == "pos" else r"$\langle N(<u)\rangle$",
                    title=f"{case['metadata']['kind']}, N={case['metadata']['n']}, seed={case['metadata']['seed']}")
        axes[0].legend(fontsize=8)
        axes[1].semilogx(fit["local_radius"], fit["local_D"], ".-")
        axes[1].set(xlabel="separation [" + unit + "]", ylabel="local count slope")
        fig.tight_layout(); fig.savefig(output / (stem + "_neighbours.pdf")); plt.close(fig)


def distribution_plot_data(values, bins=None, cdf_points=2048):
    """Full-population histogram and bounded exact-rank CDF, including rare extremes.

    CDF drawing uses at most cdf_points+64 vertices, regardless of population size.
    Its first and last points and sixteen highest ranks are retained exactly.
    """
    values = np.asarray(values)
    positive = values[values > 0]
    if not len(positive) or not np.all(np.isfinite(positive)):
        raise ValueError("Distribution plotting requires finite positive values")
    if bins is None:
        lower, upper = positive.min(), positive.max()
        if lower == upper:
            lower, upper = lower*.9, upper*1.1
        bins = np.geomspace(lower, upper, 71)
    counts, edges = np.histogram(positive, bins=bins)
    if counts.sum() != len(positive):
        raise ValueError("Histogram bounds omit full-population diffusion values")
    density = counts/(len(positive)*np.diff(edges))
    ordered = np.sort(positive)
    n = len(ordered)
    indices = np.unique(np.concatenate((
        np.linspace(0, n-1, min(n, cdf_points), dtype=int),
        np.arange(max(0, n-16), n), np.arange(min(16, n)))))
    return {"counts": counts, "density": density, "edges": edges,
            "cdf_values": ordered[indices], "cdf_probability": (indices+1)/n,
            "n_population": len(values), "n_positive": n,
            "n_cdf_vertices": len(indices), "data_scope": "full diffusion population"}


def _plot_groups(cases, output):
    import matplotlib.pyplot as plt
    groups = {}
    for case in cases:
        metadata = case["metadata"]
        parameters = json.dumps({key: metadata.get(key) for key in PLOT_PARAMETERS + ("epsilon", "v_epsilon")}, sort_keys=True)
        groups.setdefault((metadata["space"], metadata["n"], parameters), []).append(case)
    for (space, n, parameters), group in groups.items():
        populations = [case["display"]["distribution_values"] for case in group]
        positive_populations = [values[values > 0] for values in populations if np.any(values > 0)]
        if not positive_populations:
            continue
        lower = min(values.min() for values in positive_populations)
        upper = max(values.max() for values in positive_populations)
        if lower == upper:
            lower, upper = lower*.9, upper*1.1
        bins = np.geomspace(lower, upper, 71)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for index, case in enumerate(group):
            values = case["display"]["distribution_values"]
            distribution = distribution_plot_data(values, bins)
            label = f"{case['metadata']['kind']} seed={case['metadata']['seed']}"
            color = f"C{index % 10}"
            axes[0].stairs(distribution["density"], distribution["edges"], label=label, color=color)
            axes[1].plot(distribution["cdf_values"], distribution["cdf_probability"], label=label, color=color)
            axes[0].axvline(case["summary"]["raw"]["median"], alpha=.35, ls="--", color=color)
            axes[0].axvline(case["summary"]["raw"]["mean"], alpha=.35, ls=":", color=color)
        for ax in axes:
            ax.set(xscale="log", xlabel=DIFFUSION_LABEL)
        axes[0].set(ylabel="probability density"); axes[1].set(ylabel="cumulative fraction")
        axes[0].text(.03, .97, "dashed: median; dotted: mean", transform=axes[0].transAxes, va="top", fontsize=7)
        axes[1].legend(fontsize=6)
        m = group[0]["metadata"]
        fig.suptitle(f"{space}, N={n}; R0={m['R0']} kpc, theta={m.get('theta')}, veps-factor={m.get('veps_factor', 1)}\n"
                     f"axes={m.get('axes')}, Perlin scale={m.get('noise_scale')}, step={m.get('noise_step')}; full population, bounded CDF vertices", fontsize=9)
        suffix = hashlib.sha256(parameters.encode()).hexdigest()[:8]
        fig.tight_layout(); fig.savefig(output / f"{space}_N{n}_{suffix}_distribution.pdf"); plt.close(fig)
    convergence_groups = {}
    for case in cases:
        metadata = case["metadata"]
        parameters = json.dumps({key: metadata.get(key) for key in PLOT_PARAMETERS}, sort_keys=True)
        convergence_groups.setdefault((metadata["space"], parameters), []).append(case)
    for (space, parameters), group in convergence_groups.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = set()
        for case in group:
            s, m = case["summary"], case["metadata"]
            if s["zeta_median"] is None or m["kind"] == REFERENCE[m["space"]]:
                continue
            label = m["kind"]
            ax.scatter(m["n"], s["zeta_median"], label=label if label not in labels else None)
            labels.add(label)
        ax.set(xscale="log", xlabel="particle number N", ylabel=r"$\zeta$ (median)")
        m = group[0]["metadata"]
        ax.set_title(f"{space}: R0={m['R0']} kpc, theta={m.get('theta')}, veps-factor={m.get('veps_factor', 1)}\n"
                     f"axes={m.get('axes')}, Perlin scale={m.get('noise_scale')}, step={m.get('noise_step')}", fontsize=9)
        ax.axhline(1., ls="--", color="grey")
        if labels:
            ax.legend(fontsize=8)
        suffix = hashlib.sha256(parameters.encode()).hexdigest()[:8]
        fig.tight_layout(); fig.savefig(output / f"zeta_vs_N_{space}_{suffix}.pdf"); plt.close(fig)
    calibration_groups = {}
    for case in cases:
        metadata = case["metadata"]
        if metadata["kind"] == REFERENCE[metadata["space"]]:
            parameters = json.dumps({key: metadata.get(key) for key in PLOT_PARAMETERS}, sort_keys=True)
            calibration_groups.setdefault((metadata["space"], parameters), []).append(case)
    for (space, parameters), group in calibration_groups.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for seed in sorted({case["metadata"]["seed"] for case in group}):
            selected = sorted([case for case in group if case["metadata"]["seed"] == seed], key=lambda c: c["metadata"]["n"])
            ax.plot([case["metadata"]["n"] for case in selected], [case["summary"]["calibration_factor"] for case in selected],
                    ".-", label=f"seed={seed}")
        m = group[0]["metadata"]
        ax.set(xscale="log", xlabel="particle number N", ylabel=r"$C_r$" if space == "pos" else r"$C_v$")
        ax.set_title(f"Numerical reference calibration: {space}\nR0={m['R0']} kpc, theta={m.get('theta')}, veps-factor={m.get('veps_factor', 1)}", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        suffix = hashlib.sha256(parameters.encode()).hexdigest()[:8]
        fig.savefig(output / f"calibration_{space}_{suffix}.pdf"); plt.close(fig)
    fractal_groups = {}
    for case in cases:
        metadata, summary = case["metadata"], case["summary"]
        if metadata["space"] != "pos" or not summary["fractal"] or summary["fractal"].get("D_frac") is None or summary["zeta_median"] is None:
            continue
        parameters = json.dumps({key: metadata.get(key) for key in ("R0", "Rm", "v0", "mass", "theta", "eps_factor")}, sort_keys=True)
        fractal_groups.setdefault(parameters, []).append(case)
    for parameters, group in fractal_groups.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for case in group:
            s, m = case["summary"], case["metadata"]
            x, y = s["fractal"]["D_frac"], s["zeta_median"]
            ax.scatter(x, y, label=f"{m['kind']}, N={m['n']}, seed={m['seed']}")
            if m["kind"] == "perlin":
                ax.annotate(f"s={m.get('noise_scale')}, step={m.get('noise_step')}", (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6)
        ax.set(xlabel=r"measured $D_{\rm frac}$", ylabel=r"$\zeta_r$ (median)")
        ax.legend(fontsize=6)
        ax.set_title("Saved cumulative-count fit; measured points only", fontsize=9)
        fig.tight_layout()
        suffix = hashlib.sha256(parameters.encode()).hexdigest()[:8]
        fig.savefig(output / f"pos_zeta_vs_Dfrac_{suffix}.pdf"); plt.close(fig)


def combined_compatibility(position_metadata, velocity_metadata):
    """Shared physical model only; velocity smoothing belongs to the velocity marginal.

    Position metadata may record an unused v_epsilon and therefore is not required
    to share that value with velocity diagnostics. Each marginal already requires
    its own compatible numerical reference.
    """
    conflicts = []
    for key in COMBINED_PHYSICAL_FIELDS:
        if key not in position_metadata or key not in velocity_metadata:
            conflicts.append(key + ": missing")
        elif any(not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
                 for value in (position_metadata[key], velocity_metadata[key])):
            conflicts.append(key + ": invalid")
        elif not _equivalent(position_metadata[key], velocity_metadata[key]):
            conflicts.append(key + ": differs")
    return conflicts


def _distribution_values(case):
    values = case.get("display", {}).get("distribution_values")
    return values if values is not None else _load_diffusion(case)[:, 1]


def combined_median_check(cases, max_pairs=200000, issues=None):
    """Independent pair draws test the separable estimator, not a 6D DF."""
    result = []
    positions = [c for c in cases if c["metadata"]["space"] == "pos" and c["metadata"]["kind"] != "uniform" and c["summary"]["reference_cases"]]
    velocities = [c for c in cases if c["metadata"]["space"] == "vel" and c["metadata"]["kind"] != "iso" and c["summary"]["reference_cases"]]
    lookup = {c["case_id"]: c for c in cases}
    for pos in positions:
        for vel in velocities:
            if pos["metadata"]["n"] != vel["metadata"]["n"]:
                continue
            conflicts = combined_compatibility(pos["metadata"], vel["metadata"])
            if conflicts:
                if issues is not None:
                    issues.append({"position_case": pos["case_id"], "velocity_case": vel["case_id"],
                                   "issue": "Combined median skipped: incompatible physical models",
                                   "fields": conflicts})
                continue
            # Prefer the same realization seed; differing seeds remain independent.
            if any(v["metadata"]["n"] == pos["metadata"]["n"] and v["metadata"]["kind"] == vel["metadata"]["kind"]
                   and v["metadata"]["seed"] == pos["metadata"]["seed"]
                   and not combined_compatibility(pos["metadata"], v["metadata"])
                   for v in velocities) and vel["metadata"]["seed"] != pos["metadata"]["seed"]:
                continue
            seed = int.from_bytes(hashlib.sha256((pos["case_id"]+"|"+vel["case_id"]).encode()).digest()[:8], "little")
            marginals = []
            for case in (pos, vel):
                modified = _distribution_values(case)
                refs = [lookup[key] for key in case["summary"]["reference_cases"]]
                # Equal probability per reference realization, matching ensemble policy.
                reference_samples = [_distribution_values(ref) for ref in refs]
                marginals.append((modified, reference_samples))
            count = min(max_pairs, len(marginals[0][0]), len(marginals[1][0]))
            ratios, pair_seeds = [], []
            for repetition in range(5):
                trial_seed = (seed + repetition * 0x9E3779B97F4A7C15) % (1 << 64)
                pair_seeds.append(trial_seed)
                rng = np.random.default_rng(trial_seed)
                draws = []
                for modified, references in marginals:
                    modified_draw = modified[rng.choice(len(modified), count, replace=False)]
                    choice = rng.integers(0, len(references), count)
                    reference_draw = np.empty(count)
                    for i, reference in enumerate(references):
                        mask = choice == i
                        reference_draw[mask] = reference[rng.choice(len(reference), int(mask.sum()), replace=False)]
                    draws.append((modified_draw, reference_draw))
                mod_product = draws[0][0] * draws[1][0]
                ref_product = draws[0][1] * draws[1][1]
                ratios.append(float(np.median(mod_product) / np.median(ref_product)))
            product_ratio = float(np.median(ratios))
            approximation = pos["summary"]["zeta_median"] * vel["summary"]["zeta_median"]
            result.append({"position_case": pos["case_id"], "velocity_case": vel["case_id"],
                           "n_pairs": count, "pair_seed": seed,
                           "shuffle_repetitions": 5, "pair_seeds": pair_seeds,
                           "shuffle_ratios": ratios, "shuffle_ratio_range": [min(ratios), max(ratios)],
                           "product_distribution_median_ratio": product_ratio,
                           "product_of_median_enhancements": approximation,
                           "relative_difference": product_ratio/approximation-1,
                           "note": "Median and full range of five independent seeded shuffle pairs, at most 200000 pairs each; equal reference-realization weighting; pairing scatter is not realization uncertainty; separable-estimator check only"})
    return result


def realization_scatter(cases):
    """Group repeated seeds of the identical model; never pool different shapes.

    Radial controls vary angular seed at fixed original particle seed. Other
    models vary particle seed, keeping the Perlin field seed fixed. Duplicate
    copies of one seed are listed but counted only once.
    """
    groups = {}
    for case in cases:
        metadata = case["metadata"]
        parameters = {key: metadata.get(key) for key in REALIZATION_FIELDS}
        radial = metadata["space"] == "pos" and metadata["kind"] == "radial"
        if radial:
            parameters["source_particle_seed"] = metadata["seed"]
        else:
            parameters["angular_seed"] = metadata.get("angular_seed")
        key = json.dumps(clean_json(parameters), sort_keys=True)
        groups.setdefault(key, []).append(case)
    output = []
    for key, group in sorted(groups.items()):
        parameters = json.loads(key)
        radial = parameters["space"] == "pos" and parameters["kind"] == "radial"
        seed_key = "angular_seed" if radial else "seed"
        by_seed = {}
        for case in group:
            by_seed.setdefault(case["metadata"].get(seed_key), []).append(case)
        independent = [copies[0] for _, copies in sorted(by_seed.items(), key=lambda pair: str(pair[0]))]
        metrics = {}
        metric_values = {"zeta_median": [c["summary"]["zeta_median"] for c in independent],
                         "zeta_mean": [c["summary"]["zeta_mean"] for c in independent]}
        for statistic in ("median", "mean", "q99"):
            metric_values["raw_"+statistic] = [c["summary"]["raw"][statistic] for c in independent]
        metric_values["D_frac"] = [(c["summary"].get("fractal") or {}).get("D_frac") for c in independent]
        for metric, values in metric_values.items():
            valid = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
            metrics[metric] = {"n": len(valid), "status": "estimated" if len(valid) >= 2 else "not_estimated",
                               "values": values, "mean": float(valid.mean()) if len(valid) else None,
                               "median": float(np.median(valid)) if len(valid) else None,
                               "range": [float(valid.min()), float(valid.max())] if len(valid) else None,
                               "sample_standard_deviation": float(valid.std(ddof=1)) if len(valid) >= 2 else None}
        output.append({"parameters": parameters, "status": "estimated" if len(independent) >= 2 else "not_estimated",
                       "n_cases": len(group), "n_realizations": len(independent), "varying_seed": seed_key,
                       "realizations": [{"seed": seed, "case_ids": [c["case_id"] for c in copies]}
                                        for seed, copies in sorted(by_seed.items(), key=lambda pair: str(pair[0]))],
                       "metrics": metrics,
                       "interpretation": ("Angular-randomization scatter at fixed original Perlin particle sample; not independent source samples"
                                          if radial else "Particle-sampling scatter at fixed generation parameters and fixed Perlin field seed; not scatter between independent Perlin fields"),
                       "uncertainty_note": "Singletons have no scatter estimate. Repeated copies of a seed count once. Small realization counts do not establish precise confidence intervals."})
    return output


def analyze_root(root, output, make_plots=True):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases, issues = [], []
    for path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("schema") != "mfrt_trace_v2":
            continue
        if metadata.get("legacy_GXX_correction", False):
            raise ValueError(f"New schema cannot request or contain legacy GXX correction: {path}")
        if metadata.get("complete") is not True:
            issues.append({"case": str(path.parent.relative_to(root)),
                           "issue": "Incomplete metadata: case skipped"})
            continue
        for key in ("space", "kind", "n", "seed", "R0", "Rm", "v0", "D_classical_median"):
            if key not in metadata:
                raise ValueError(f"Missing metadata field {key}: {path}")
        if metadata["space"] not in REFERENCE:
            raise ValueError(f"Unknown space: {path}")
        metadata.setdefault("crossing_length", metadata["R0"])
        metadata.setdefault("t_cross", metadata["crossing_length"]/metadata["v0"])
        metadata.setdefault("velocity_second_moment", metadata["v0"]**2)
        cases.append({"directory": path.parent, "metadata": metadata, "case_id": str(path.parent.relative_to(root))})
    if not cases:
        raise ValueError(f"No mfrt_trace_v2 cases found below {root}")
    cases.sort(key=lambda c: c["metadata"]["kind"] != REFERENCE[c["metadata"]["space"]])
    for case in cases:
        refs, mode = match_references(case, cases)
        if not refs:
            issues.append({"case": case["case_id"], "issue": "No compatible numerical reference; calibrated statistics unavailable"})
        summary, error, display = _case_analysis(case, refs, mode)
        case.update(summary=summary, tree_error=error, display=display)
        if mode == "other_seed_reference_ensemble" and refs:
            issues.append({"case": case["case_id"], "issue": "No same-seed reference; ensemble of compatible reference realizations used"})
        if make_plots:
            _plot_case(case, output)
    report = {"schema": "mfrt_trace_analysis_v2", "scalar": "Dtot=Tr(D)",
              "legacy_GXX_correction_applied": False,
              "calibration": "One reference-median factor applied to all particles and statistics; enhancement ratios use matching statistics",
              "cases": [case["summary"] for case in cases], "issues": issues,
              "combined_median_check": combined_median_check(cases, issues=issues),
              "realization_scatter": realization_scatter(cases)}
    errors = {case["case_id"]: case["tree_error"] for case in cases if case["tree_error"]}
    (output / "summary.json").write_text(json.dumps(clean_json(report), indent=2, allow_nan=False)+"\n")
    (output / "tree_errors.json").write_text(json.dumps(clean_json(errors), indent=2, allow_nan=False)+"\n")
    fields = ["case_id", "space", "kind", "n", "seed", "calibration_factor", "zeta_median", "zeta_mean", "D_frac"]
    fields += [prefix+key for prefix in ("raw_", "calibrated_") for key in ("median", "mean", "q16", "q84", "q95", "q99")]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            s = case["summary"]
            row = {key: s.get(key, case["metadata"].get(key)) for key in fields}
            row["D_frac"] = s["fractal"].get("D_frac") if s["fractal"] else None
            for prefix in ("raw", "calibrated"):
                for key in ("median", "mean", "q16", "q84", "q95", "q99"):
                    row[prefix+"_"+key] = s[prefix][key] if s[prefix] else None
            writer.writerow(clean_json(row))
    if make_plots:
        _plot_groups(cases, output)
    return clean_json(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    report = analyze_root(args.root, args.output, not args.no_plots)
    print(f"Analyzed {len(report['cases'])} cases; {len(report['issues'])} reference notices; output={args.output}")


if __name__ == "__main__":
    main()
