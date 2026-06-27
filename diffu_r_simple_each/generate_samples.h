#ifndef GENERATE_SAMPLES_H
#define GENERATE_SAMPLES_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>           // For OpenMP parallelization
#include "PerlinNoise.hpp" // Include the siv::PerlinNoise header
#include "diffu.h"

void Perlin_motion(
    std::vector<PositionSpace>& positions, PrecisionSetting scale = 0.1f, int octaves = 4,
    PrecisionSetting step_size = 0.5f, unsigned int seed = 123456u,
    PrecisionSetting persistence = 0.5f, PrecisionSetting lacunarity = 2.0f
);
void Perlin_motion(
    std::vector<VelocitySpace>& positions, PrecisionSetting scale = 0.1f, int octaves = 4,
    PrecisionSetting step_size = 0.5f, unsigned int seed = 123456u,
    PrecisionSetting persistence = 0.5f, PrecisionSetting lacunarity = 2.0f
);
void write_samples_to_binary(const std::vector<PositionSpace>& samples, const std::string& filename);
void read_binary_and_convert_to_txt(const std::string& binary_file, const std::string& txt_file, std::vector<PositionSpace>& positions, bool is_update_samples=false);

void generate_spherical_samples(size_t N_particles, std::vector<PositionSpace>& samples);
void generate_and_save_samples(size_t N_particles, PrecisionSetting radius, std::vector<PositionSpace>& samples, const std::string& binary_file, const std::string& txt_file);
void generate_noised_samples(std::vector<PositionSpace>& positions, int N_iter=14);
void generate_noised_samples(
    std::vector<PositionSpace>& positions, int N_iter, PrecisionSetting scale, int octaves,
    PrecisionSetting step_size, unsigned int seed = 123456u,
    PrecisionSetting persistence = 0.5f, PrecisionSetting lacunarity = 2.0f
);
void generate_noised_samples(std::vector<VelocitySpace>& positions, int N_iter=14);
void generate_noised_samples(
    std::vector<VelocitySpace>& positions, int N_iter, PrecisionSetting scale, int octaves,
    PrecisionSetting step_size, unsigned int seed = 123456u,
    PrecisionSetting persistence = 0.5f, PrecisionSetting lacunarity = 2.0f
);

/*  Function to randomly select unique targets without replacement.
*/
template <typename TYPEN>
std::vector<TYPEN> select_random_targets(
    const std::vector<TYPEN>& vel, double rate_calculate_input
){
    size_t N = vel.size();
    if (N == 0) {
        throw std::runtime_error("Input velocity sample is empty.");
    }

    double rate_calculate = rate_calculate_input;
    if (rate_calculate_input <= 0.0 || rate_calculate_input > 1.0) {
        std::cerr<<"Invalid rate_calculate. Must be in (0, 1]. Set it to 1.\n";
        rate_calculate = 1.0;
    }

    size_t N_targets = static_cast<size_t>(rate_calculate * N);

    // Generate a list of indices and shuffle
    std::vector<size_t> indices(N);
    for (size_t i = 0; i < N; ++i) indices[i] = i;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    // Select the first N_targets elements
    std::vector<TYPEN> vel_targets;
    vel_targets.reserve(N_targets);
    for (size_t i = 0; i < N_targets; ++i) {
        vel_targets.push_back(vel[indices[i]]);
    }

    return vel_targets;
}

/*  Sample N velocity vectors from a Maxwellian (isotropic) distribution,
    i.e. each component is drawn from a Normal distribution with mean 0 and std dev sigma.
*/
std::vector<VelocitySpace> sample_maxwellian_isotropic(size_t N, PrecisionSetting sigma);

/*  Logistic (blending) function: g(v)=1/(1+exp(-k*(v-v1))).
*/
PrecisionSetting g_func(PrecisionSetting v, PrecisionSetting k, PrecisionSetting v1);

/*  Evaluate the combined 1D speed probability density function:
    f_comb(v) = 4π*v² [ f_G(v)*(1-g(v)) + A_pl*f_pl(v)*g(v) ]
    where:
    f_G(v) = (2πσ²)^(-1.5) * exp(-0.5*(v/σ)²)
    f_pl(v) = (A_pl / vc³) * (ε + (v/vc)²)^(-γ)
*/
PrecisionSetting df_velocity(
    PrecisionSetting v, PrecisionSetting sigma,  PrecisionSetting vc, PrecisionSetting gamma, 
    PrecisionSetting k, PrecisionSetting v1, PrecisionSetting A_pl, PrecisionSetting epsilon
);

/*  Sample an isotropic velocity DF using rejection sampling on the 1D speed PDF.
    Once a candidate speed is accepted, a random direction is drawn (from independent normals),
    and the candidate speed is assigned along that direction.
*/
std::vector<VelocitySpace> sample_velocity_isotropic_df(size_t N,
                                                        PrecisionSetting sigma, 
                                                        PrecisionSetting vc, 
                                                        PrecisionSetting gamma,
                                                        PrecisionSetting k,
                                                        PrecisionSetting v1,
                                                        PrecisionSetting A_pl,
                                                        PrecisionSetting epsilon,
                                                        PrecisionSetting vmax);

/*  Transform an isotropic velocity sample (given as a vector of VelocitySpace) 
    into an anisotropic one by reassigning directions according to the specified anisotropic scaling.
    sigma_3 must have three elements.
*/
std::vector<VelocitySpace> sample_isotropic_to_anisotropic(const std::vector<VelocitySpace>& velocities,
                                                           const std::vector<PrecisionSetting>& sigma_3);

/*  Evaluate an anisotropic DF at (vx,vy,vz) given separate dispersion parameters.
    Here, sigma and vc are vectors of length 3.
*/
PrecisionSetting df_anisotropic(PrecisionSetting vx, PrecisionSetting vy, PrecisionSetting vz,
                                const std::vector<PrecisionSetting>& sigma, 
                                const std::vector<PrecisionSetting>& vc, 
                                PrecisionSetting gamma,
                                PrecisionSetting k,
                                PrecisionSetting v1,
                                PrecisionSetting A_pl,
                                PrecisionSetting epsilon);

/*  Sample an anisotropic velocity DF (in 3D) using rejection sampling.
    vmax_three is a vector of three maximum values for each velocity component.
*/
std::vector<VelocitySpace> sample_anisotropic_df(size_t N,
                                                 const std::vector<PrecisionSetting>& sigma, 
                                                 const std::vector<PrecisionSetting>& vc, 
                                                 PrecisionSetting gamma,
                                                 PrecisionSetting k,
                                                 PrecisionSetting v1,
                                                 PrecisionSetting A_pl,
                                                 PrecisionSetting epsilon,
                                                 const std::vector<PrecisionSetting>& vmax_three);

void generate_simple_xv(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_generate, PrecisionSetting mean_radius, 
    const std::vector<PrecisionSetting>& mean_3, const std::vector<PrecisionSetting>& dispersion_3
);
void generate_AA_xv( //unfinished
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_generate, PrecisionSetting rs, PrecisionSetting vs, 
    PrecisionSetting* powerlaw, PrecisionSetting* coef
);
void generate_samples_Jeans( //unfinished
    std::vector<VelocitySpace>& vel, PrecisionSetting* coef
);
void change_xv_to_multi_big_clusters(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, 
    int n_clusters, PrecisionSetting k_translate=1.f, PrecisionSetting k_shrink=0.5f
);

#endif
