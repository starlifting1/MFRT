#ifndef GENERATE_SAMPLES_H
#define GENERATE_SAMPLES_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>           // For OpenMP parallelization
#include "PerlinNoise.hpp" // Include the siv::PerlinNoise header
#include "diffu.h"

void Perlin_motion(const std::vector<PositionSpace>& positions, float scale = 0.1f, int octaves = 4, float step_size = 0.5f);
void write_samples_to_binary(const std::vector<PositionSpace>& samples, const std::string& filename);
void read_binary_and_convert_to_txt(const std::string& binary_file, const std::string& txt_file, std::vector<PositionSpace>& positions, bool is_update_samples=false);
void generate_spherical_samples(size_t N_particles, std::vector<PositionSpace>& samples);
void generate_and_save_samples(size_t N_particles, float radius, std::vector<PositionSpace>& samples, const std::string& binary_file, const std::string& txt_file);
void generate_noised_samples(std::vector<PositionSpace>& positions, int N_iter=14);

void generate_simple_xv(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_generate, float mean_radius, 
    const std::vector<float>& mean_3, const std::vector<float>& dispersion_3
);
void generate_AA_xv(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_generate, float rs, float vs, float* powerlaw, float* coef
);
void change_xv_to_multi_big_clusters(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, 
    int n_clusters, float k_translate=1.f, float k_shrink=0.5f
);

#endif