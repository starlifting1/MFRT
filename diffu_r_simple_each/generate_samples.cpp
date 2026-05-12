#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream> // For file I/O
#include <omp.h>  // For OpenMP parallelization
#include "generate_samples.h"



// Function to write samples into a binary file
void write_samples_to_binary(const std::vector<PositionSpace>& samples, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }

    for (const auto& sample : samples) {
        file.write(reinterpret_cast<const char*>(&sample), sizeof(PositionSpace));
    }

    file.close();
    std::cout << "Samples written to binary file: " << filename << std::endl;
}

// Function to read samples from a binary file, populate a vector, and convert to a text file
void read_binary_and_convert_to_txt(
    const std::string& binary_file, const std::string& txt_file, 
    std::vector<PositionSpace>& positions, bool is_update_samples
){
    std::ifstream bin_file(binary_file, std::ios::binary);
    if (!bin_file) {
        std::cerr << "Error: Cannot open file " << binary_file << " for reading." << std::endl;
        return;
    }

    std::ofstream txt_file_out(txt_file);
    if (!txt_file_out) {
        std::cerr << "Error: Cannot open file " << txt_file << " for writing." << std::endl;
        return;
    }

    // Ensure the positions vector has the correct size
    if (positions.size() == 0 && is_update_samples) {
        std::cerr << "Error: The positions vector size must be non-zero. Exit." << std::endl;
        exit(0);
    }

    // Read binary data into both the positions vector and text file
    PositionSpace sample;
    size_t index = 0;
    while (bin_file.read(reinterpret_cast<char*>(&sample), sizeof(PositionSpace))) {
        // Save to the positions vector
        if (index < positions.size() && is_update_samples) {
            positions[index] = sample;
            ++index;
        }

        // Save to the text file
        txt_file_out << sample.x << "\t" << sample.y << "\t" << sample.z << " \n";
    }

    bin_file.close();
    txt_file_out.close();
    std::cout << "Binary data converted to text file: " << txt_file << std::endl;

    // Ensure all data was read into the vector
    if (index != positions.size()) {
        std::cerr << "Warning: Fewer particles read than expected. Expected " << positions.size() << ", but got " << index << "." << std::endl;
    }
}

// Function to generate uniform distribution with spherical shape
void generate_spherical_samples(size_t N_particles, std::vector<PositionSpace>& samples){
    PrecisionSetting radius = 1.f;
    // std::vector<PositionSpace> samples(N_particles);
    samples.resize(N_particles);
    std::random_device rd;
    // std::mt19937 gen(39); //debug
    std::mt19937 gen(rd()); //random by device
    std::uniform_real_distribution<PrecisionSetting> u(0.0f, 1.0f);
    std::uniform_real_distribution<PrecisionSetting> v(0.0f, 1.0f);

#pragma omp parallel for
    for (size_t i = 0; i < N_particles; ++i) {
        PrecisionSetting theta = 2.0f * M_PI * u(gen); // Uniform azimuthal angle
        PrecisionSetting phi = std::acos(1.0f - 2.0f * u(gen)); // Uniform polar angle
        PrecisionSetting r = std::cbrt(v(gen)) * radius; // Uniform volume distribution

        samples[i] = {r * std::sin(phi) * std::cos(theta),
                      r * std::sin(phi) * std::sin(theta),
                      r * std::cos(phi)};
    }
}



// Perlin motion function for PositionSpace
void Perlin_motion(
    std::vector<PositionSpace>& positions, PrecisionSetting scale, int octaves, PrecisionSetting step_size
){
    // Initialize the Perlin noise generator
    siv::PerlinNoise perlin(123456u); // Seed for consistency

    // Parallel loop for PositionSpace updates
#pragma omp parallel for
    for (size_t i = 0; i < N_particles; ++i) {
        const PositionSpace& pos = positions[i];
        PrecisionSetting x = pos.x, y = pos.y, z = pos.z;

        PrecisionSetting noise_dx = 0.0f, noise_dy = 0.0f, noise_dz = 0.0f;

        // Generate multi-layer Perlin noise by iterating through octaves
        for (int octave = 0; octave < octaves; ++octave) {
            PrecisionSetting frequency = std::pow(2.0f, octave); // Increasing frequency for each octave
            PrecisionSetting amplitude = std::pow(0.5f, octave); // Decreasing amplitude for each octave

            noise_dx += amplitude * perlin.noise3D(x * scale * frequency, y * scale * frequency, z * scale * frequency);
            noise_dy += amplitude * perlin.noise3D(y * scale * frequency, z * scale * frequency, x * scale * frequency);
            noise_dz += amplitude * perlin.noise3D(z * scale * frequency, x * scale * frequency, y * scale * frequency);
        }

        // Update the position
        positions[i] = {x + noise_dx * step_size, y + noise_dy * step_size, z + noise_dz * step_size};
    }
}

void generate_noised_samples(std::vector<PositionSpace>& positions, int N_iter){
    std::cout<<"pos: "<<positions[0].x<<"\n";
    
    // Parameters for Perlin motion
    // int N_iter = 0;
    // int N_iter = 14;
    // // int N_iter = 20;
    int octaves = 4;
    // PrecisionSetting scale = 0.05;
    PrecisionSetting scale = 0.1;
    // // PrecisionSetting step_size = 1.0;
    PrecisionSetting step_size = 2.0;
    // PrecisionSetting step_size = 4.0;

    // Apply Perlin motion for the given number of iterations
    for (int i = 0; i < N_iter; ++i) {
        Perlin_motion(positions, scale, octaves, step_size);
    }
    std::cout<<"pos: "<<positions[0].x<<"\n";
}



// Perlin motion function for VelocitySpace
void Perlin_motion(
    std::vector<VelocitySpace>& positions, PrecisionSetting scale, int octaves, PrecisionSetting step_size
){
    // Initialize the Perlin noise generator
    siv::PerlinNoise perlin(123456u); // Seed for consistency

    // Parallel loop for PositionSpace updates
#pragma omp parallel for
    for (size_t i = 0; i < N_particles; ++i) {
        const VelocitySpace& pos = positions[i];
        PrecisionSetting x = pos.x, y = pos.y, z = pos.z;

        PrecisionSetting noise_dx = 0.0f, noise_dy = 0.0f, noise_dz = 0.0f;

        // Generate multi-layer Perlin noise by iterating through octaves
        for (int octave = 0; octave < octaves; ++octave) {
            PrecisionSetting frequency = std::pow(2.0f, octave); // Increasing frequency for each octave
            PrecisionSetting amplitude = std::pow(0.5f, octave); // Decreasing amplitude for each octave

            noise_dx += amplitude * perlin.noise3D(x * scale * frequency, y * scale * frequency, z * scale * frequency);
            noise_dy += amplitude * perlin.noise3D(y * scale * frequency, z * scale * frequency, x * scale * frequency);
            noise_dz += amplitude * perlin.noise3D(z * scale * frequency, x * scale * frequency, y * scale * frequency);
        }

        // Update the position
        positions[i] = {x + noise_dx * step_size, y + noise_dy * step_size, z + noise_dz * step_size};
    }
}

void generate_noised_samples(std::vector<VelocitySpace>& positions, int N_iter){
    std::cout<<"pos: "<<positions[0].x<<"\n";
    
    // Parameters for Perlin motion
    // int N_iter = 0;
    // int N_iter = 14;
    // // int N_iter = 20;
    int octaves = 4;
    PrecisionSetting scale = 0.02;
    PrecisionSetting step_size = 20.0;

    // Apply Perlin motion for the given number of iterations
    for (int i = 0; i < N_iter; ++i) {
        Perlin_motion(positions, scale, octaves, step_size);
    }
    std::cout<<"pos: "<<positions[0].x<<"\n";
}



/*  To generate random uniform positions DF and Gaussian velocities DF.
*/
void generate_simple_xv(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_generate, PrecisionSetting mean_radius, 
    const std::vector<PrecisionSetting>& mean_3, const std::vector<PrecisionSetting>& dispersion_3
){
    pos.resize(N_generate);
    vel.resize(N_generate);

    std::random_device rd;
    // std::mt19937 gen(39); //debug
    std::mt19937 gen(rd()); //random by device

    std::uniform_real_distribution<PrecisionSetting> u(0.0f, 1.0f);
    std::uniform_real_distribution<PrecisionSetting> v(0.0f, 1.0f);

    std::normal_distribution<PrecisionSetting> dist_x(mean_3[0], dispersion_3[0]); // Gaussian for vx
    std::normal_distribution<PrecisionSetting> dist_y(mean_3[1], dispersion_3[1]); // Gaussian for vy
    std::normal_distribution<PrecisionSetting> dist_z(mean_3[2], dispersion_3[2]); // Gaussian for vz

#pragma omp parallel for
    for (size_t i = 0; i < N_generate; ++i) {
        PrecisionSetting theta = 2.0f * M_PI * u(gen); // Uniform azimuthal angle
        PrecisionSetting phi = std::acos(1.0f - 2.0f * u(gen)); // Uniform polar angle
        PrecisionSetting r = std::cbrt(v(gen)); // Uniform volume distribution
        pos[i].x = r * std::sin(phi) * std::cos(theta);
        pos[i].y = r * std::sin(phi) * std::sin(theta);
        pos[i].z = r * std::cos(phi);

        vel[i].x = dist_x(gen);
        vel[i].y = dist_y(gen);
        vel[i].z = dist_z(gen);
    }

    readjust_positions(pos, mean_radius);
    readjust_velocities_v2m_sqrt(vel, mean_3, dispersion_3);
    return ;
}

void generate_samples_Jeans(
    std::vector<VelocitySpace>& vel, PrecisionSetting* coef
){
    // by Jeans equation constraints
    return ;
}

void generate_AA_xv(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_generate, PrecisionSetting rs, PrecisionSetting vs, PrecisionSetting* powerlaw, PrecisionSetting* coef
){
    // by angles and actions
    return ;
}

void change_xv_to_multi_big_clusters(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, 
    int n_clusters, PrecisionSetting k_translate, PrecisionSetting k_shrink
){
    // ?? move files, upload, rename diffu, plot much
    int n_clusters_ = 0;
    if(!(n_clusters==2 && n_clusters==4 && n_clusters==6)){
        std::cerr<<"The @param n_clusters can be only one of {} in this function. Using default n_clusters=2\n";
        n_clusters_ = 2;
    }else{
        n_clusters_ = n_clusters;
    }

    size_t N_genrate = pos.size();
    PrecisionSetting GM_unit = 1.f;
    translate_to_center(pos);
    PrecisionSetting R_typ = get_mean_radius(pos);
    PrecisionSetting Phi_mean_esitimate = -GM_unit/R_typ;
    PrecisionSetting Ek_mean = 0.5f*calculate_quadratic_mean(vel);
    PrecisionSetting Etot_mean = Phi_mean_esitimate+Ek_mean;
    PrecisionSetting frac_modify = abs(Etot_mean)/abs(Ek_mean);
    print_scalars(Ek_mean, Phi_mean_esitimate);

    std::vector<std::vector<PrecisionSetting>> directions;
    directions.resize(n_clusters_);
    if(n_clusters_==2){ //seperate on x direction to 2 clusters
        directions[0] = {-1.f, 0.f, 0.f}; //x- to reference point
        directions[1] = {1.f, 0.f, 0.f}; //x+ to reference point
    }else if(n_clusters_==4){ //seperate on x,y direction to 4 clusters
        directions[0] = {-1.f, 0.f, 0.f}; //x- to reference point
        directions[1] = {1.f, 0.f, 0.f}; //x+ to reference point
        directions[2] = {0.f, -1.f, 0.f}; //y- to reference point
        directions[3] = {0.f, 1.f, 0.f}; //y+ to reference point
    }else if(n_clusters_==6){ //seperate on x,y,z direction to 6 clusters
        directions[0] = {-1.f, 0.f, 0.f}; //x- to reference point
        directions[1] = {1.f, 0.f, 0.f}; //x+ to reference point
        directions[2] = {0.f, -1.f, 0.f}; //y- to reference point
        directions[3] = {0.f, 1.f, 0.f}; //y+ to reference point
        directions[4] = {0.f, 0.f, -1.f}; //z- to reference point
        directions[5] = {0.f, 0.f, 1.f}; //z+ to reference point
    }
    PrecisionSetting frac_vel = std::sqrt(1.f/(k_translate+k_shrink));

    for(size_t i=0;i<N_genrate;++i){
        pos[i].x = k_shrink*pos[i].x+k_translate*R_typ*directions[i%n_clusters_][0];
        pos[i].y = k_shrink*pos[i].y+k_translate*R_typ*directions[i%n_clusters_][1];
        pos[i].z = k_shrink*pos[i].z+k_translate*R_typ*directions[i%n_clusters_][2];

        vel[i].x *= frac_vel;
        vel[i].y *= frac_vel;
        vel[i].z *= frac_vel;
    }
    translate_to_center(pos);
    std::cout<<"Seperate xv into several clusters, done.\n";
    return ;
}



// Initialize a random number generator (C++11 style)
// static 
std::mt19937 rng(std::random_device{}());

// Helper function: sample from a normal distribution with mean 0 and std dev sigma.
// static 
PrecisionSetting rand_normal(PrecisionSetting sigma) {
    std::normal_distribution<PrecisionSetting> dist(0.0, sigma);
    return dist(rng);
}

// Helper function: sample uniformly from [a, b].
// static 
PrecisionSetting rand_uniform(PrecisionSetting a, PrecisionSetting b) {
    std::uniform_real_distribution<PrecisionSetting> dist(a, b);
    return dist(rng);
}

// -----------------------------------------------------------------
// each component is sampled from N(0, sigma).
// -----------------------------------------------------------------
std::vector<VelocitySpace> sample_maxwellian_isotropic(size_t N, PrecisionSetting sigma) {
    std::vector<VelocitySpace> samples;
    samples.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        VelocitySpace v;
        v.x = rand_normal(sigma);
        v.y = rand_normal(sigma);
        v.z = rand_normal(sigma);
        samples.push_back(v);
    }
    return samples;
}

// -----------------------------------------------------------------
// g_func: logistic blending function: g(v)=1/(1+exp(-k*(v-v1))).
// -----------------------------------------------------------------
PrecisionSetting g_func(PrecisionSetting v, PrecisionSetting k, PrecisionSetting v1) {
    return 1.0 / (1.0 + std::exp(-k * (v - v1)));
}

// -----------------------------------------------------------------
// df_velocity: Evaluate the combined 1D speed PDF.
// -----------------------------------------------------------------
PrecisionSetting df_velocity(PrecisionSetting v,
                               PrecisionSetting sigma, 
                               PrecisionSetting vc, 
                               PrecisionSetting gamma,
                               PrecisionSetting k,
                               PrecisionSetting v1,
                               PrecisionSetting A_pl,
                               PrecisionSetting epsilon) {
    // Gaussian core term.
    PrecisionSetting factor_G = std::pow(2.0 * M_PI * sigma * sigma, -1.5);
    PrecisionSetting f_G = factor_G * std::exp(-0.5 * std::pow(v/sigma, 2));
    
    // Power-law tail term (translated by epsilon).
    PrecisionSetting f_pl = A_pl / std::pow(vc, 3) * std::pow(epsilon + std::pow(v/vc, 2), -gamma);
    
    // Blending function.
    PrecisionSetting g = g_func(v, k, v1);
    
    // Combined PDF with spherical volume element.
    PrecisionSetting f_combined = 4.0 * M_PI * v*v * (f_G * (1 - g) + f_pl * g);
    return f_combined;
}

// -----------------------------------------------------------------
// sample_velocity_isotropic_df: Rejection sampling on the 1D speed PDF
// then assign an isotropic direction.
// -----------------------------------------------------------------
std::vector<VelocitySpace> sample_velocity_isotropic_df(size_t N,
                                                        PrecisionSetting sigma, 
                                                        PrecisionSetting vc, 
                                                        PrecisionSetting gamma,
                                                        PrecisionSetting k,
                                                        PrecisionSetting v1,
                                                        PrecisionSetting A_pl,
                                                        PrecisionSetting epsilon,
                                                        PrecisionSetting vmax) {
    std::vector<VelocitySpace> samples;
    samples.reserve(N);
    
    // Create a grid of speeds to determine f_max.
    const size_t gridCount = 1000;
    std::vector<PrecisionSetting> v_grid(gridCount);
    PrecisionSetting dv = vmax / (gridCount - 1);
    for (size_t i = 0; i < gridCount; ++i) {
        v_grid[i] = i * dv;
    }
    PrecisionSetting f_max = 0.0;
    for (size_t i = 0; i < gridCount; ++i) {
        PrecisionSetting f_val = df_velocity(v_grid[i], sigma, vc, gamma, k, v1, A_pl, epsilon);
        if (f_val > f_max)
            f_max = f_val;
    }
    
    // Rejection sampling for speed.
    while (samples.size() < N) {
        PrecisionSetting v_candidate = rand_uniform(0.0, vmax);
        PrecisionSetting u = rand_uniform(0.0, f_max);
        if (u > df_velocity(v_candidate, sigma, vc, gamma, k, v1, A_pl, epsilon))
            continue;
        // Sample an isotropic random direction.
        PrecisionSetting x = rand_normal(1.0);
        PrecisionSetting y = rand_normal(1.0);
        PrecisionSetting z = rand_normal(1.0);
        PrecisionSetting norm = std::sqrt(x*x + y*y + z*z);
        if (norm == 0) continue;
        x /= norm; y /= norm; z /= norm;
        VelocitySpace v;
        v.x = v_candidate * x;
        v.y = v_candidate * y;
        v.z = v_candidate * z;
        samples.push_back(v);
    }
    
    return samples;
}

// -----------------------------------------------------------------
// sample_isotropic_to_anisotropic: Transform an isotropic sample into an anisotropic one,
// by reassigning the direction with different scaling per axis given by sigma_3.
// -----------------------------------------------------------------
std::vector<VelocitySpace> sample_isotropic_to_anisotropic(const std::vector<VelocitySpace>& velocities,
                                                           const std::vector<PrecisionSetting>& sigma_3) {
    if (sigma_3.size() != 3)
        throw std::runtime_error("sigma_3 vector must have 3 elements.");
    std::vector<VelocitySpace> v_aniso;
    v_aniso.reserve(velocities.size());
    
    for (size_t i = 0; i < velocities.size(); ++i) {
        // Compute the candidate speed from the isotropic sample.
        PrecisionSetting v_mag = std::sqrt(velocities[i].x*velocities[i].x +
                                           velocities[i].y*velocities[i].y +
                                           velocities[i].z*velocities[i].z);
        // Generate a random direction.
        PrecisionSetting dx = rand_normal(1.0);
        PrecisionSetting dy = rand_normal(1.0);
        PrecisionSetting dz = rand_normal(1.0);
        PrecisionSetting norm = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (norm == 0) continue;
        dx /= norm; dy /= norm; dz /= norm;
        
        // Scale the direction by the anisotropic dispersion factors.
        PrecisionSetting scaled_x = sigma_3[0] * dx;
        PrecisionSetting scaled_y = sigma_3[1] * dy;
        PrecisionSetting scaled_z = sigma_3[2] * dz;
        PrecisionSetting scaled_norm = std::sqrt(scaled_x*scaled_x + scaled_y*scaled_y + scaled_z*scaled_z);
        if (scaled_norm == 0) continue;
        scaled_x /= scaled_norm;
        scaled_y /= scaled_norm;
        scaled_z /= scaled_norm;
        
        VelocitySpace v;
        v.x = v_mag * scaled_x;
        v.y = v_mag * scaled_y;
        v.z = v_mag * scaled_z;
        v_aniso.push_back(v);
    }
    
    return v_aniso;
}

// -----------------------------------------------------------------
// df_anisotropic: Evaluate the anisotropic DF at (vx,vy,vz).
// The Gaussian core uses per-axis dispersions (sigma vector) and the tail uses per-axis scales (vc vector).
// -----------------------------------------------------------------
PrecisionSetting df_anisotropic(PrecisionSetting vx, PrecisionSetting vy, PrecisionSetting vz,
                                const std::vector<PrecisionSetting>& sigma, 
                                const std::vector<PrecisionSetting>& vc, 
                                PrecisionSetting gamma,
                                PrecisionSetting k,
                                PrecisionSetting v1,
                                PrecisionSetting A_pl,
                                PrecisionSetting epsilon) {
    if (sigma.size() != 3 || vc.size() != 3)
        throw std::runtime_error("sigma and vc must have 3 elements each.");
    
    // Compute the speed.
    PrecisionSetting v = std::sqrt(vx*vx + vy*vy + vz*vz);
    
    // Compute normalized squared terms for the Gaussian.
    PrecisionSetting v_fracto_sigma_sq = (vx / sigma[0])*(vx / sigma[0])
                                        + (vy / sigma[1])*(vy / sigma[1])
                                        + (vz / sigma[2])*(vz / sigma[2]);
    // Similarly for the tail.
    PrecisionSetting v_fracto_vc_sq = (vx / vc[0])*(vx / vc[0])
                                    + (vy / vc[1])*(vy / vc[1])
                                    + (vz / vc[2])*(vz / vc[2]);
    // Products for normalization.
    PrecisionSetting sigma_mult = sigma[0] * sigma[1] * sigma[2];
    PrecisionSetting vc_mult = vc[0] * vc[1] * vc[2];
    
    // Gaussian core.
    PrecisionSetting f_G = std::pow(2.0 * M_PI, -1.5) / sigma_mult * std::exp(-0.5 * v_fracto_sigma_sq);
    // Power-law tail.
    PrecisionSetting f_pl = A_pl / vc_mult * std::pow(epsilon + v_fracto_vc_sq, -gamma);
    // Blending: use scalar speed v for g(v)
    PrecisionSetting g = g_func(v, k, v1);
    // Combined DF.
    PrecisionSetting f_comb = f_G * (1 - g) + f_pl * g;
    
    return f_comb;
}

// -----------------------------------------------------------------
// sample_anisotropic_df: Rejection sampling in 3D for an anisotropic DF.
// Candidate velocities are sampled uniformly from a box defined by vmax_three (vector of length 3).
// -----------------------------------------------------------------
std::vector<VelocitySpace> sample_anisotropic_df(size_t N,
                                                 const std::vector<PrecisionSetting>& sigma, 
                                                 const std::vector<PrecisionSetting>& vc, 
                                                 PrecisionSetting gamma,
                                                 PrecisionSetting k,
                                                 PrecisionSetting v1,
                                                 PrecisionSetting A_pl,
                                                 PrecisionSetting epsilon,
                                                 const std::vector<PrecisionSetting>& vmax_three) {
    if (vmax_three.size() != 3)
        throw std::runtime_error("vmax_three must have 3 elements.");
    
    std::vector<VelocitySpace> samples;
    samples.reserve(N);
    
    // Determine a simple f_max from the candidate (using (0,0,0)); note that the maximum may not be exactly at zero.
    PrecisionSetting f_max = df_anisotropic(0.0, 0.0, 0.0, sigma, vc, gamma, k, v1, A_pl, epsilon);
    // You might want to determine a better f_max by scanning a grid.
    
    while (samples.size() < N) {
        VelocitySpace candidate;
        candidate.x = rand_uniform(-vmax_three[0], vmax_three[0]);
        candidate.y = rand_uniform(-vmax_three[1], vmax_three[1]);
        candidate.z = rand_uniform(-vmax_three[2], vmax_three[2]);
        
        PrecisionSetting prob = df_anisotropic(candidate.x, candidate.y, candidate.z, sigma, vc, gamma, k, v1, A_pl, epsilon);
        PrecisionSetting u = rand_uniform(0.0, f_max);
        if (u > prob)
            continue;
        samples.push_back(candidate);
    }
    
    return samples;
}
