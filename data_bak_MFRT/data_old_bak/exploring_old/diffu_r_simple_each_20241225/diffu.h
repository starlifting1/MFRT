#ifndef DIFFU_H
#define DIFFU_H

#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <iostream>
#include <type_traits>



// Utility function to print a single vector
template <typename T>
void print_vector(const std::vector<T>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

// Variadic template to print multiple vectors
template <typename First, typename... Rest>
void print_vectors(const First& first, const Rest&... rest) {
    if constexpr (std::is_same_v<std::decay_t<First>, std::vector<typename First::value_type>>) {
        print_vector(first);
    } else {
        std::cout << first;
    }
    if constexpr (sizeof...(rest) > 0) {
        std::cout << ", ";
        print_vectors(rest...);
    }
}

// // Variadic template function to print multiple vectors
// template <typename... Args>
// void print_vectors(const Args&... args) {
//     (print_vector(args), ...); // Fold expression to call print_vector for each argument
// }

// Function to print a single scalar
template <typename T>
void print_scalar(const T& scalar) {
    std::cout << scalar << "\n";
}

// Variadic template function to print multiple scalars
template <typename... Args>
void print_scalars(const Args&... args) {
    (print_scalar(args), ...); // Fold expression to call print_scalar for each argument
}

// // Variadic template using fold expressions //in C++23
// template <typename... Args>
// void print_vectors(const Args&... args) {
//     ((std::cout << (std::ranges::range<Args> ? print_vector(args), "" : args) << ", "), ...);
//     std::cout << "\b\b "; // Remove trailing comma
// }



// Structure to represent a particle
#define PrecisionSetting float
// #define PrecisionSetting double

struct PositionSpace {
    PrecisionSetting x, y, z; // Particle positions
};

struct VelocitySpace {
    PrecisionSetting x, y, z; // Particle velocities
};

struct xvSpaceCartesian {
    PrecisionSetting Pos[3];
    PrecisionSetting Vel[3];
};

struct Diffu_info {
    size_t N_particles;
    PrecisionSetting Diffu_0, Diffu_mean, Diffu_median;
    PrecisionSetting Diffu_min, Diffu_max;
};

const PrecisionSetting G = 43007.1f;
const PrecisionSetting frac_mass = 1.0f; //frac mass of stars component of galaxy
// const PrecisionSetting frac_mass = 0.05f; //frac mass of stars component of galaxy
const PrecisionSetting lambda1 = 3.0f;
const PrecisionSetting lambda2 = 0.2f;

extern PrecisionSetting M_total; //6.85f = 137.0f*0.05f;
extern PrecisionSetting R0; //50.0f;
extern PrecisionSetting v0;
extern size_t N_particles;
extern size_t N_samples;
extern PrecisionSetting b90;
extern PrecisionSetting coef_const_Diffu;

extern std::vector<PrecisionSetting> mean_3; // = {0.f, 0.f, 0.f}; //user setting
extern std::vector<PrecisionSetting> dispersion_3; // = {60.f, 60.f, 60.f}; //user setting



// Function prototypes
inline PrecisionSetting calculate_b90() {
    return R0 / (lambda2 * N_particles);
}

inline PrecisionSetting calculate_v0() {
    PrecisionSetting M_total_gal = M_total/frac_mass;
    return sqrt(G*M_total_gal/R0);
}

inline PrecisionSetting calculate_coef_const_Diffu(){
    return lambda1 * G * G * M_total * M_total / (v0 * R0 * N_particles * N_samples);
}

bool read_xv_from_binary(const std::string& file_path, std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_particles);
bool save_Diffur_to_binary(const std::string& filename, const std::vector<PrecisionSetting>& D_direct, const std::vector<PrecisionSetting>& D_tree);
bool save_Diffur_to_txt(const std::string& filename, const std::vector<PrecisionSetting>& D_direct, const std::vector<PrecisionSetting>& D_tree);
bool readParticles_from_txt(const std::string& file_path, std::vector<PositionSpace>& particles, size_t N_particles);
void directSummation(const std::vector<PositionSpace>& particles, std::vector<PrecisionSetting>& D_direct, size_t N_particles);
void write_xv_to_binary(const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, const std::string& filename);

std::vector<PrecisionSetting> get_center(const std::vector<PositionSpace>& pos);
std::vector<PrecisionSetting> get_center(const std::vector<VelocitySpace>& velocities);
PrecisionSetting get_mean_radius(const std::vector<PositionSpace>& particles, bool is_consider_center=false);
PrecisionSetting get_mean_radius(const std::vector<VelocitySpace>& particles, bool is_consider_center=false);
PrecisionSetting calculate_quadratic_mean(const std::vector<PositionSpace>& velocities);
PrecisionSetting calculate_quadratic_mean(const std::vector<VelocitySpace>& velocities);
void translate_to_center(std::vector<PositionSpace>& particles);
void translate_to_center(std::vector<VelocitySpace>& particles);
void readjust_positions(std::vector<PositionSpace>& particles, PrecisionSetting r_mean);
void readjust_velocities(std::vector<VelocitySpace>& particles, const std::vector<PrecisionSetting>& mean_3, const std::vector<PrecisionSetting>& dispersion_3);

PrecisionSetting diffusion_reference_value_cylinder(PrecisionSetting R_minus=-1., bool is_using_formula=true);
PrecisionSetting calculateDiffusionCoefficient_DS(const std::vector<PositionSpace>& particles, const PositionSpace& target, size_t index_target);
Diffu_info calculate_diffu_statistics(const std::vector<PrecisionSetting>& data, std::string file_statistics, bool is_save=true);
void print_all_particles(const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, size_t print_interval=1);
void print_particles_info(const std::vector<PositionSpace>& particles);
void print_particles_info(const std::vector<VelocitySpace>& particles);
void print_galaxy_setting_info();
void print_diffu_info(const Diffu_info& Diffu_statistics);

#endif
