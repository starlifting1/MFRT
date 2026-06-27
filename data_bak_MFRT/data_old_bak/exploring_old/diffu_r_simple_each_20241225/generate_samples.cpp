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
    float radius = 1.f;
    // std::vector<PositionSpace> samples(N_particles);
    samples.resize(N_particles);
    std::random_device rd;
    // std::mt19937 gen(rd()); //random by device
    std::mt19937 gen(39); //debug
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_real_distribution<float> v(0.0f, 1.0f);

#pragma omp parallel for
    for (size_t i = 0; i < N_particles; ++i) {
        float theta = 2.0f * M_PI * u(gen); // Uniform azimuthal angle
        float phi = std::acos(1.0f - 2.0f * u(gen)); // Uniform polar angle
        float r = std::cbrt(v(gen)) * radius; // Uniform volume distribution

        samples[i] = {r * std::sin(phi) * std::cos(theta),
                      r * std::sin(phi) * std::sin(theta),
                      r * std::cos(phi)};
    }
}



// Perlin motion function
void Perlin_motion(
    std::vector<PositionSpace>& positions, float scale, int octaves, float step_size
){
    // // Output vector for updated positions
    // const size_t N_particles = positions.size();
    // std::vector<PositionSpace> new_positions(N_particles);

    // Initialize the Perlin noise generator
    siv::PerlinNoise perlin(123456u); // Seed for consistency

    // Parallel loop for PositionSpace updates
#pragma omp parallel for
    for (size_t i = 0; i < N_particles; ++i) {
        const PositionSpace& pos = positions[i];
        float x = pos.x, y = pos.y, z = pos.z;

        float noise_dx = 0.0f, noise_dy = 0.0f, noise_dz = 0.0f;

        // Generate multi-layer Perlin noise by iterating through octaves
        for (int octave = 0; octave < octaves; ++octave) {
            float frequency = std::pow(2.0f, octave); // Increasing frequency for each octave
            float amplitude = std::pow(0.5f, octave); // Decreasing amplitude for each octave

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
    // const size_t N_particles = positions.size();
    // const float rs = 1.0f;
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_real_distribution<float> dis(0.0f, rs * 2.0f);
    // for (size_t i = 0; i < N_particles; ++i) {
    //     positions[i] = {dis(gen), dis(gen), dis(gen)}; //debug
    // }
    
    // Parameters for Perlin motion
    // int N_iter = 0;
    // int N_iter = 14;
    // // int N_iter = 20;
    int octaves = 4;
    // float scale = 0.05f;
    float scale = 0.1f;
    // // float step_size = 1.0f;
    float step_size = 2.0f;
    // float step_size = 4.0f;

    // Apply Perlin motion for the given number of iterations
    for (int i = 0; i < N_iter; ++i) {
        Perlin_motion(positions, scale, octaves, step_size);
        // positions = Perlin_motion(positions, scale, octaves, step_size);
    }
    std::cout<<"pos: "<<positions[0].x<<"\n";
}

/*  To generate random uniform positions DF and Gaussian velocities DF.
*/
void generate_simple_xv(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_generate, float mean_radius, 
    const std::vector<float>& mean_3, const std::vector<float>& dispersion_3
){
    pos.resize(N_generate);
    vel.resize(N_generate);

    std::random_device rd;
    // std::mt19937 gen(39); //debug
    std::mt19937 gen(rd()); //random by device

    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_real_distribution<float> v(0.0f, 1.0f);

    std::normal_distribution<float> dist_x(mean_3[0], dispersion_3[0]); // Gaussian for vx
    std::normal_distribution<float> dist_y(mean_3[1], dispersion_3[1]); // Gaussian for vy
    std::normal_distribution<float> dist_z(mean_3[2], dispersion_3[2]); // Gaussian for vz

#pragma omp parallel for
    for (size_t i = 0; i < N_generate; ++i) {
        float theta = 2.0f * M_PI * u(gen); // Uniform azimuthal angle
        float phi = std::acos(1.0f - 2.0f * u(gen)); // Uniform polar angle
        float r = std::cbrt(v(gen)); // Uniform volume distribution
        pos[i].x = r * std::sin(phi) * std::cos(theta);
        pos[i].y = r * std::sin(phi) * std::sin(theta);
        pos[i].z = r * std::cos(phi);

        vel[i].x = dist_x(gen);
        vel[i].y = dist_y(gen);
        vel[i].z = dist_z(gen);
    }

    readjust_positions(pos, mean_radius);
    readjust_velocities(vel, mean_3, dispersion_3);
    return ;
}

void generate_AA_xv(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_generate, float rs, float vs, float* powerlaw, float* coef
){
    // by angles and actions
    return ;
}

void change_xv_to_multi_big_clusters(
    std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, 
    int n_clusters, float k_translate, float k_shrink
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
    float GM_unit = 1.f;
    translate_to_center(pos);
    float R_typ = get_mean_radius(pos);
    float Phi_mean_esitimate = -GM_unit/R_typ;
    float Ek_mean = 0.5f*calculate_quadratic_mean(vel);
    float Etot_mean = Phi_mean_esitimate+Ek_mean;
    float frac_modify = abs(Etot_mean)/abs(Ek_mean);
    print_scalars(Ek_mean, Phi_mean_esitimate);

    std::vector<std::vector<float>> directions;
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
    float frac_vel = sqrt(1.f/(k_translate+k_shrink));

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
