#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

#include "diffu.h"
#include "fractal_dim.h"
#include "generate_samples.h"
#include "Tree_method.h"

struct PerlinSweepParam {
    PrecisionSetting step_size;
    PrecisionSetting scale;
    int iterations;
    int octaves;
    PrecisionSetting persistence;
    PrecisionSetting lacunarity;
    unsigned int seed;
};

struct DiffuSummary {
    size_t valid_count = 0;
    size_t skipped_count = 0;
    PrecisionSetting mean = 0.0;
    PrecisionSetting median = 0.0;
    PrecisionSetting stddev = 0.0;
    PrecisionSetting min = 0.0;
    PrecisionSetting max = 0.0;
    PrecisionSetting q05 = 0.0;
    PrecisionSetting q16 = 0.0;
    PrecisionSetting q50 = 0.0;
    PrecisionSetting q84 = 0.0;
    PrecisionSetting q95 = 0.0;
};

static std::string timestamp_string() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&t);
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return out.str();
}

static std::string run_label(size_t run_id) {
    std::ostringstream out;
    out << "run_" << std::setw(3) << std::setfill('0') << run_id;
    return out.str();
}

static std::vector<PrecisionSetting> rescale_dispersion_keep_ratio_quiet(
    const std::vector<PrecisionSetting>& mean_3,
    const std::vector<PrecisionSetting>& ratio,
    PrecisionSetting v2m_sqrt_target
) {
    return rescale_dispersion_keep_ratio(mean_3, ratio, v2m_sqrt_target);
}

static void generate_simple_xv_seeded(
    std::vector<PositionSpace>& pos,
    std::vector<VelocitySpace>& vel,
    size_t N_generate,
    PrecisionSetting mean_radius,
    const std::vector<PrecisionSetting>& mean_3,
    const std::vector<PrecisionSetting>& dispersion_3,
    unsigned int seed
) {
    pos.resize(N_generate);
    vel.resize(N_generate);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<PrecisionSetting> u(0.0, 1.0);
    std::uniform_real_distribution<PrecisionSetting> v(0.0, 1.0);
    std::normal_distribution<PrecisionSetting> dist_x(mean_3[0], dispersion_3[0]);
    std::normal_distribution<PrecisionSetting> dist_y(mean_3[1], dispersion_3[1]);
    std::normal_distribution<PrecisionSetting> dist_z(mean_3[2], dispersion_3[2]);

    for (size_t i = 0; i < N_generate; ++i) {
        PrecisionSetting theta = 2.0 * M_PI * u(gen);
        PrecisionSetting phi = std::acos(1.0 - 2.0 * u(gen));
        PrecisionSetting r = std::cbrt(v(gen));
        pos[i].x = r * std::sin(phi) * std::cos(theta);
        pos[i].y = r * std::sin(phi) * std::sin(theta);
        pos[i].z = r * std::cos(phi);

        vel[i].x = dist_x(gen);
        vel[i].y = dist_y(gen);
        vel[i].z = dist_z(gen);
    }

    readjust_positions(pos, mean_radius);
    readjust_velocities_v2m_sqrt(vel, mean_3, dispersion_3);
}

static PrecisionSetting quantile_sorted(const std::vector<PrecisionSetting>& sorted, PrecisionSetting q) {
    if (sorted.empty()) {
        return 0.0;
    }
    PrecisionSetting index = (sorted.size() - 1) * q;
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));
    if (lower == upper) {
        return sorted[lower];
    }
    PrecisionSetting weight = index - lower;
    return (1.0 - weight) * sorted[lower] + weight * sorted[upper];
}

static DiffuSummary summarize_diffu(const std::vector<PrecisionSetting>& diffu) {
    DiffuSummary summary;
    std::vector<PrecisionSetting> valid;
    valid.reserve(diffu.size());

    for (const auto value : diffu) {
        if (std::isfinite(value)) {
            valid.push_back(value);
        } else {
            ++summary.skipped_count;
        }
    }

    summary.valid_count = valid.size();
    if (valid.empty()) {
        return summary;
    }

    std::sort(valid.begin(), valid.end());
    summary.min = valid.front();
    summary.max = valid.back();
    summary.mean = std::accumulate(valid.begin(), valid.end(), 0.0) / valid.size();

    PrecisionSetting variance = 0.0;
    for (const auto value : valid) {
        PrecisionSetting delta = value - summary.mean;
        variance += delta * delta;
    }
    summary.stddev = std::sqrt(variance / valid.size());

    summary.q05 = quantile_sorted(valid, 0.05);
    summary.q16 = quantile_sorted(valid, 0.16);
    summary.q50 = quantile_sorted(valid, 0.50);
    summary.q84 = quantile_sorted(valid, 0.84);
    summary.q95 = quantile_sorted(valid, 0.95);
    summary.median = summary.q50;
    return summary;
}

static std::vector<PrecisionSetting> calculate_tree_diffu(const std::vector<PositionSpace>& pos) {
    std::vector<PrecisionSetting> diffu(pos.size(), 0.0);
    Octree tree;
    tree.build(pos);

    size_t N_display = std::max<size_t>(1, pos.size() / 10);
#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < pos.size(); ++i) {
        diffu[i] = tree.calculateDiffusionCoefficient(pos[i]);
        if (i % N_display == 0) {
            std::cout << "Diffu: " << i << ": " << std::setprecision(10) << diffu[i] << "\n";
        }
    }
    return diffu;
}

static void write_diffu_txt(const std::string& filename, const std::vector<PrecisionSetting>& diffu) {
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot open diffusion output file: " + filename);
    }
    out << "# id\tDiffu\n";
    for (size_t i = 0; i < diffu.size(); ++i) {
        out << (i + 1) << "\t" << std::setprecision(17) << diffu[i] << "\n";
    }
}

static void write_sample_txt(
    const std::string& filename,
    const std::vector<PositionSpace>& pos,
    const std::vector<VelocitySpace>& vel
) {
    if (pos.size() != vel.size()) {
        throw std::runtime_error("Position and velocity sample sizes do not match.");
    }

    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot open sample output file: " + filename);
    }
    out << "# id\tx\ty\tz\tvx\tvy\tvz\n";
    for (size_t i = 0; i < pos.size(); ++i) {
        out << (i + 1) << "\t" << std::setprecision(17)
            << pos[i].x << "\t" << pos[i].y << "\t" << pos[i].z << "\t"
            << vel[i].x << "\t" << vel[i].y << "\t" << vel[i].z << "\n";
    }
}

static void write_summary_row(
    std::ofstream& out,
    size_t run_id,
    const std::string& sample_type,
    const PerlinSweepParam& param,
    size_t N_iter,
    size_t N,
    PrecisionSetting Rm_current,
    const std::string& sample_file,
    const std::string& diffu_file,
    const std::string& nr_file,
    const DiffuSummary& summary,
    const DiffuSummary& reference
) {
    PrecisionSetting erdc_median = reference.median != 0.0 ? summary.median / reference.median : 0.0;
    PrecisionSetting erdc_mean = reference.mean != 0.0 ? summary.mean / reference.mean : 0.0;

    out << std::setprecision(17) << run_id << "\t"
        << summary.median << "\t"
        << sample_type << "\t"
        << N << "\t"
        << N_iter << "\t"
        << std::setprecision(17) << (param.step_size / 2.0) << "\t"
        << param.scale << "\t"
        << param.octaves << "\t"
        << param.persistence << "\t"
        << param.lacunarity << "\t"
        << param.step_size << "\t"
        << summary.mean << "\t"
        << summary.stddev << "\t"
        << summary.min << "\t"
        << summary.max << "\t"
        << summary.q05 << "\t"
        << summary.q16 << "\t"
        << summary.q50 << "\t"
        << summary.q84 << "\t"
        << summary.q95 << "\t"
        << erdc_mean << "\t"
        << erdc_median << "\t"
        << sample_file << "\t"
        << diffu_file << "\t"
        << nr_file << "\t"
        << param.seed << "\t"
        << Rm_current << "\t"
        << summary.valid_count << "\t"
        << summary.skipped_count << "\n";
}

int main(int argc, char* argv[]) {

    //// settings
    // Keep particle and target counts fixed for all reference and Perlin samples.
    const size_t N_requested = 40000;
    // const size_t N_requested = 40000;
    PrecisionSetting M_total_gal = 137.0;
    PrecisionSetting R0_setting = 50.0;
    size_t max_runs = 0;

    if (argc > 1) {
        std::stringstream(argv[1]) >> max_runs;
    }
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [max_perlin_runs]\n";
        return 1;
    }

    N_particles = N_requested;
    N_samples = N_particles;
    M_total = M_total_gal * frac_mass;
    R0 = R0_setting;
    calculate_fixed_const_coefs();
    print_galaxy_setting_info("pos_fractal_sweep init");

    vmean_3 = {0.0, 0.0, 0.0};
    dispersion_3_iso_ratio = {1.0, 1.0, 1.0};
    dispersion_3_high_ratio = {8.0, 6.0, 1.0};
    std::vector<PrecisionSetting> dispersion_3_iso =
        rescale_dispersion_keep_ratio_quiet(vmean_3, dispersion_3_iso_ratio, v2m_sqrt);

    std::vector<PositionSpace> pos_ref;
    std::vector<VelocitySpace> vel_ref;
    unsigned int sample_seed = 123456u;
    generate_simple_xv_seeded(pos_ref, vel_ref, N_particles, Rm, vmean_3, dispersion_3_iso, sample_seed);
    readjust_positions(pos_ref, Rm);
    const PrecisionSetting reference_Rm = get_mean_radius(pos_ref);
    std::cout << "reference mean_radius = " << reference_Rm << "\n";
    const double rate_calculate = 1.e4 / N_particles;

    //// files
    const std::filesystem::path executable_dir = getExecutablePath();
    std::filesystem::path base_dir = executable_dir /
        ".." / "data" / "samples_pos" / "pos_fractal_sweep" / ("run_" + timestamp_string());
    std::filesystem::path per_particle_dir = base_dir / "per_particle";
    std::filesystem::path samples_dir = base_dir / "samples";
    std::filesystem::path nr_counts_dir = base_dir / "nr_counts";
    std::filesystem::create_directories(per_particle_dir);
    std::filesystem::create_directories(samples_dir);
    std::filesystem::create_directories(nr_counts_dir);

    std::filesystem::path summary_file = base_dir / "summary.txt";
    std::ofstream summary_out(summary_file);
    if (!summary_out) {
        std::cerr << "Cannot open summary file: " << summary_file << "\n";
        return 1;
    }

    summary_out
        << "# run_id\tDiffu_median\tsample_type\tN\tN_iter\tnoise_strength\tnoise_scale\toctaves\tpersistence\tlacunarity\t"
        << "step_size\tDiffu_mean\tDiffu_stddev\tDiffu_min\tDiffu_max\tDiffu_q05\tDiffu_q16\tDiffu_q50\tDiffu_q84\tDiffu_q95\t"
        << "ERDC_mean\tERDC_median\tsample_file\tdiffu_file\tnr_file\tseed\tRm\tvalid_count\tskipped_count\n";

    PerlinSweepParam reference_param{0.0, 0.1, 0, 4, 0.5, 2.0, 123456u};
    std::cout << "Calculating reference diffusion.\n";
    std::vector<PrecisionSetting> diffu_ref = calculate_tree_diffu(pos_ref);
    DiffuSummary reference_summary = summarize_diffu(diffu_ref);

    std::filesystem::path ref_sample_file = samples_dir / "run_000_reference_positions.txt";
    std::filesystem::path ref_diffu_file = per_particle_dir / "run_000_reference_diffu.txt";
    std::filesystem::path ref_nr_file = nr_counts_dir / "nr_pos_run_000_reference.txt";
    calculate_frac_dim(pos_ref, reference_Rm, ref_nr_file.string(), rate_calculate);
    write_sample_txt(ref_sample_file.string(), pos_ref, vel_ref);
    write_diffu_txt(ref_diffu_file.string(), diffu_ref);
    write_summary_row(
        summary_out, 0, "reference", reference_param, 0, N_particles, reference_Rm,
        std::filesystem::relative(ref_sample_file, executable_dir).string(),
        std::filesystem::relative(ref_diffu_file, executable_dir).string(),
        std::filesystem::relative(ref_nr_file, executable_dir).string(),
        reference_summary, reference_summary
    );

    //// sweep
    // Edit these vectors to tune the sweep. 
    // The legacy paper-like configuration is scale=0.1, step_size=2.0,
    //\ iterations=14, octaves=4, persistence=0.5, lacunarity=2.0.
    // std::vector<PrecisionSetting> scales = {0.05, 0.1, 0.2};
    std::vector<PrecisionSetting> scales = {0.5, 0.1, 0.2};
    // std::vector<PrecisionSetting> step_sizes = {1., 2.0, 3.};
    std::vector<PrecisionSetting> step_sizes = {1., 2., 3., 5.};
    // std::vector<int> iteration_counts = {0, 7, 14, 21};
    std::vector<int> iteration_counts = {0, 7, 35, 70, 105};
    std::vector<int> octave_counts = {4};
    std::vector<PrecisionSetting> persistences = {0.5};
    std::vector<PrecisionSetting> lacunarities = {2.0};
    std::vector<unsigned int> seeds = {123456u};
    std::vector<PerlinSweepParam> grid;
    for (const auto scale : scales) {
        for (const auto step_size : step_sizes) {
            for (const auto N_iter : iteration_counts) {
                for (const auto octaves : octave_counts) {
                    for (const auto persistence : persistences) {
                        for (const auto lacunarity : lacunarities) {
                            for (const auto seed : seeds) {
                                grid.push_back({step_size, scale, N_iter, octaves, persistence, lacunarity, seed});
                            }
                        }
                    }
                }
            }
        }
    }

    size_t n_runs = max_runs > 0 ? std::min(max_runs, grid.size()) : grid.size();
    for (size_t i = 0; i < n_runs; ++i) {
        size_t run_id = i + 1;
        const auto& param = grid[i];
        const int N_iter = param.iterations;
        std::cout << "Running sweep " << run_id << "/" << n_runs
                  << ": step_size=" << param.step_size
                  << ", scale=" << param.scale
                  << ", iterations=" << N_iter << "\n";

        std::vector<PositionSpace> pos;
        std::vector<VelocitySpace> vel;
        copy_xv_sample(pos_ref, pos);
        copy_xv_sample(vel_ref, vel);

        if (N_iter > 0 && param.step_size != 0.0) {
            generate_noised_samples(
                pos, N_iter, param.scale, param.octaves, param.step_size,
                param.seed, param.persistence, param.lacunarity
            );
        }
        readjust_positions(pos, reference_Rm);
        const PrecisionSetting current_Rm = get_mean_radius(pos);
        std::cout << "mean_radius = " << current_Rm << "\n";

        std::vector<PrecisionSetting> diffu = calculate_tree_diffu(pos);
        DiffuSummary summary = summarize_diffu(diffu);

        std::string label = run_label(run_id);
        std::ostringstream suffix;
        suffix << label
               << "_step_" << std::fixed << std::setprecision(2) << param.step_size
               << "_scale_" << std::setprecision(2) << param.scale
               << "_iter_" << N_iter
               << "_oct_" << param.octaves
               << "_seed_" << param.seed;

        //// write
        std::filesystem::path sample_file = samples_dir / (suffix.str() + "_positions.txt");
        std::filesystem::path diffu_file = per_particle_dir / (suffix.str() + "_diffu.txt");
        std::filesystem::path nr_file = nr_counts_dir / ("nr_pos_" + suffix.str() + ".txt");
        calculate_frac_dim(pos, current_Rm, nr_file.string(), rate_calculate);
        write_sample_txt(sample_file.string(), pos, vel);
        write_diffu_txt(diffu_file.string(), diffu);

        write_summary_row(
            summary_out, run_id, "perlin", param, N_iter, N_particles, current_Rm,
            std::filesystem::relative(sample_file, executable_dir).string(),
            std::filesystem::relative(diffu_file, executable_dir).string(),
            std::filesystem::relative(nr_file, executable_dir).string(),
            summary, reference_summary
        );
        summary_out.flush();
    }

    std::cout << "Position fractal sweep complete.\n";
    std::cout << "Summary file: " << summary_file << "\n";
    std::cout << "Per-particle files: " << per_particle_dir << "\n";
    std::cout << "Sample files: " << samples_dir << "\n";
    std::cout << "Neighbor-count files: " << nr_counts_dir << "\n";
    return 0;
}
