#ifndef DIFFU_H
#define DIFFU_H

#include <vector>
#include <string>
#include <cmath>
#include <iomanip> // for std::setprecision
#include <unistd.h> // for exe path
#include <stdexcept>
#include <type_traits>
#include <iostream>
#include <cstddef>
#include <algorithm>
#include <numeric>



// #define PrecisionSetting float
#define PrecisionSetting double

// constant variables
const PrecisionSetting G = 43007.1;
const PrecisionSetting frac_mass = 1.0; //frac mass of stars component of galaxy
// const PrecisionSetting frac_mass = 0.05; //frac mass of stars component of galaxy

const PrecisionSetting lambda1_old = 3.0;
const PrecisionSetting lambda2_old = 0.2;
const PrecisionSetting lambda1 = 0.75;
const PrecisionSetting lambda2 = 1.5;
const PrecisionSetting lambda3 = 1.5;
const PrecisionSetting lambda4 = 0.2;

extern PrecisionSetting M_total; //6.85 = 137.0*0.05;
extern size_t N_particles;
extern size_t N_samples;

extern PrecisionSetting R0; //50.0;
extern PrecisionSetting Rm; //R0/lambda2
extern PrecisionSetting b90;
extern PrecisionSetting R_epsilon;
extern PrecisionSetting coef_const_Diffu;

extern PrecisionSetting v0; //the typical speed of subject particles, here one sets v0 equals to v2m_sqrt
extern PrecisionSetting v2m; //about Ek
extern PrecisionSetting v2m_sqrt;
extern PrecisionSetting v_epsilon;
extern PrecisionSetting coef_const_Diffu_vel;

extern PrecisionSetting n0; //a urepresentative niform number density
extern PrecisionSetting ln_Lambda; //the Coulomb index
extern PrecisionSetting GXX; //G_X/X for iso in BT08
extern PrecisionSetting ratio_Diffu_old;
extern PrecisionSetting ratio_Diffu_old_vel;

extern PrecisionSetting Diffu_0_unit; //unit
extern PrecisionSetting Diffu_ref;
extern PrecisionSetting coef_Diffu_parallel_iso; //this multiplies Rm*Rm*I is Diffu_parallel
extern PrecisionSetting coef_Diffu_tensor_uniform; //this multiplies v2m_sqrt*J_tensor is Diffu_uniform_tensor
extern PrecisionSetting coef_Diffu_separable; //this multiplies Rm*Rm*I and v2m_sqrt*J_tensor is Diffu_separable

extern std::vector<PrecisionSetting> vmean_3; // = {0., 0., 0.}; //user setting
extern std::vector<PrecisionSetting> dispersion_3_iso_ratio;
extern std::vector<PrecisionSetting> dispersion_3_high_ratio; // = {60., 60., 60.}; //user setting

extern std::vector<PrecisionSetting> qpers_display;



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
        std::cout << first<< ", ";
    }
    if constexpr (sizeof...(rest) > 0) {
        print_vectors(rest...);
    }
}

template <typename T>
bool check_value_consistant_rate(
    int is_continue_running, PrecisionSetting x, PrecisionSetting x0, T info, PrecisionSetting epsilon=1e-3
){
    if( std::abs(x/x0-1.)>epsilon ){ //not consist
        if(is_continue_running>1){ //print info e.g. 10, or not print info e.g. 1
            std::cout<<"Wrong for "<<info<<":\n";
            std::cout<<x<<", "<<x0<<"\n";
            // //() Note that abs() is for integer, it is not std::abs() in cmath or math.h
            // std::cout<<abs(x/x0-1.)<<", "<<abs(x0/x-1.)<<"\n";
        }
        if(is_continue_running<=0){ //exit e.g. 0
            exit(0);
        }
        return 0;
    }else{ //consist
        return 1;
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



// structs
struct Cartesian3DSpace {
    PrecisionSetting x, y, z; // Particle 3D
};

struct PositionSpace: Cartesian3DSpace{}; // Particle positions

struct VelocitySpace: Cartesian3DSpace{}; // Particle velocities

struct xvSpaceCartesian {
    PrecisionSetting Pos[3];
    PrecisionSetting Vel[3];
};

/*  The diffusion coef tensor for velocity space comparasion, fixing the position space information. 
    Here only 6 components is independent bcause $D_{i,j} \equiv D_{j,i}$ because of symmetry.
*/
struct Diffu_tensor_vel {
    PrecisionSetting D[3][3];
};
// std::vector<Diffu_tensor_vel> D_vel_direct, D_vel_tree; //resize to N size, each 3*3;

struct Diffu_info {
    size_t N_particles;
    PrecisionSetting Diffu_ref, Diffu_mean, Diffu_median;
    PrecisionSetting Diffu_min, Diffu_max, Diffu_0_big;
};

struct Statistics_info {
    size_t N_particles;
    PrecisionSetting Diffu_ref, Diffu_0_big;
    PrecisionSetting Diffu_mean, Diffu_median, Diffu_min, Diffu_max;
    Diffu_tensor_vel Diffu_vel_mean, Diffu_vel_median, Diffu_vel_min, Diffu_vel_max;
};



// Function prototypes
inline PrecisionSetting calculate_b90() {
    return R0 / (lambda4 * N_particles);
}

inline PrecisionSetting calculate_v0() {
    PrecisionSetting M_total_gal = M_total/frac_mass;
    return std::sqrt(G*M_total_gal/R0);
}

inline PrecisionSetting calculate_coef_const_Diffu(){
    return lambda1_old * G * G * M_total * M_total / (v0 * R0 * N_particles * N_samples);
}

PrecisionSetting calculate_v0_vel(const std::vector<VelocitySpace>& particles);

inline PrecisionSetting calculate_v_epsilon() {
    return v2m_sqrt / (lambda4 * N_particles);
}

inline PrecisionSetting calculate_coef_const_Diffu_vel(){
    // setting density $n0 = \frac{N}{4\pi R_0^3}$ without 3. because of crossing cylidrical space
    return G*G * M_total*M_total * log(lambda4 * N_particles) / (R0*R0*R0 * N_particles * N_samples);
}

PrecisionSetting diffusion_reference_value_Diffu_ref(); //same with extern variable Diffu_ref
void calculate_fixed_const_coefs();
PrecisionSetting diffusion_reference_value_cylinder(PrecisionSetting R_minus=-1., bool is_using_formula=true);
Diffu_info calculate_diffu_statistics(const std::vector<PrecisionSetting>& data, std::string file_statistics, bool is_save=true);
Statistics_info calculate_diffu_statistics(
    const std::vector<Diffu_tensor_vel>& diffu, std::vector<PrecisionSetting>& diffu_eff
);

void initialize_one_tensor(Diffu_tensor_vel& tensor);
void initialize_each_tensor(std::vector<Diffu_tensor_vel>& D, size_t N);
std::vector<PrecisionSetting> get_center(const std::vector<PositionSpace>& pos);
std::vector<PrecisionSetting> get_center(const std::vector<VelocitySpace>& velocities);
PrecisionSetting get_mean_radius(const std::vector<PositionSpace>& particles, bool is_consider_center=false);
PrecisionSetting get_mean_radius(const std::vector<VelocitySpace>& particles, bool is_consider_center=false);
PrecisionSetting calculate_quadratic_mean(const std::vector<PositionSpace>& velocities);
PrecisionSetting calculate_quadratic_mean(const std::vector<VelocitySpace>& velocities);
PrecisionSetting calculate_speed_square_mean(const std::vector<PrecisionSetting>& vm, const std::vector<PrecisionSetting>& dispersion_diag);
PrecisionSetting calculate_dispersion_iso(PrecisionSetting v2m_sqrt, const std::vector<PrecisionSetting>& vm);
PrecisionSetting norm_vector(const std::vector<PrecisionSetting>& vec);
std::vector<PrecisionSetting> rescale_dispersion_keep_ratio(
    const std::vector<PrecisionSetting>& vmean_three_fixed, const std::vector<PrecisionSetting>& sigma_three_old, 
    PrecisionSetting v2m_sqrt_target
);
void translate_to_center(std::vector<PositionSpace>& particles);
void translate_to_center(std::vector<VelocitySpace>& particles);
void readjust_positions(std::vector<PositionSpace>& particles, PrecisionSetting r_mean);
void readjust_velocities_v2m_sqrt(std::vector<VelocitySpace>& particles, const std::vector<PrecisionSetting>& vmean_3, const std::vector<PrecisionSetting>& dispersion_3);
void readjust_velocities_dispersion_diag(
    std::vector<VelocitySpace>& velocities, const std::vector<PrecisionSetting>& vmean_3,
    const std::vector<PrecisionSetting>& dispersion_3
);
std::vector<PrecisionSetting> np_percentile(const std::vector<PrecisionSetting>& a, const std::vector<PrecisionSetting>& q);

bool read_xv_from_binary(const std::string& file_path, std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_particles);
bool readParticles_from_txt(const std::string& file_path, std::vector<PositionSpace>& particles, size_t N_particles);
bool load_xv_from_txt(std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, const std::string& filename);
void write_xv_to_binary(const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, const std::string& filename);
void write_xv_to_txt(const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, const std::string& filename);
void copy_xv_sample(const std::vector<PositionSpace>& p_ori, std::vector<PositionSpace>& p_another);
void copy_xv_sample(const std::vector<VelocitySpace>& p_ori, std::vector<VelocitySpace>& p_another);
bool save_Diffur_to_binary(const std::string& filename, const std::vector<PrecisionSetting>& D_direct, const std::vector<PrecisionSetting>& D_tree);
bool save_Diffur_to_txt(const std::string& filename, const std::vector<PrecisionSetting>& D_direct, const std::vector<PrecisionSetting>& D_tree);
bool read_Diffur_from_binary(const std::string& filename, std::vector<PrecisionSetting>& D_direct, std::vector<PrecisionSetting>& D_tree);
bool save_percentile_to_txt(const std::string& filename, const std::vector<PrecisionSetting>& q, const std::vector<PrecisionSetting>& a);
bool save_inside_counts(const std::string& filename, const std::vector<PrecisionSetting>& radii, const std::vector<PrecisionSetting>& inscs);
bool save_all_diffu_tensors_to_binary(const std::string& filename, const std::vector<Diffu_tensor_vel>& diffu);
bool save_all_diffu_tensors_to_txt(const std::string& filename, const std::vector<Diffu_tensor_vel>& diffu);
bool read_diffu_tensors_from_binary(const std::string& filename, std::vector<Diffu_tensor_vel>& diffu);
bool save_diffu_statistics(const std::string& filename, const Statistics_info& sta);

void print_all_particles(const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, size_t print_interval=1);
void print_particles_info(const std::vector<PositionSpace>& particles);
void print_particles_info(const std::vector<VelocitySpace>& particles);
void print_galaxy_setting_info(std::string descrp="descrp");
void print_Diffu_tensor_vel(const Diffu_tensor_vel& tensor);
void print_diffu_info(const Diffu_info& Diffu_statistics);

std::string getExecutablePath();

#endif
