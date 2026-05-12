#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "PerlinNoise.hpp" // Include the siv::PerlinNoise header
#include <omp.h>           // For OpenMP parallelization

// Define a 3D vector for positions using float precision
struct Particle {
    float x, y, z;
};

// Perlin_motion function
std::vector<Particle> Perlin_motion(const std::vector<Particle>& positions, float scale = 0.1f, int octaves = 4, float step_size = 0.5f) {
    // Number of particles
    const size_t N_particles = positions.size();

    // Output vector for updated positions
    std::vector<Particle> new_positions(N_particles);

    // Initialize the Perlin noise generator
    siv::PerlinNoise perlin(123456u); // Seed for consistency

    // Parallel loop for particle updates
// #pragma omp parallel for
    for (size_t i = 0; i < N_particles; ++i) {
        const Particle& pos = positions[i];
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
        new_positions[i] = {x + noise_dx * step_size, y + noise_dy * step_size, z + noise_dz * step_size};
    }

    return new_positions;
}

int main() {
    // Define the number of particles (size_t allows large numbers)
    const size_t N_particles = static_cast<size_t>(1e7); // Example: 1 billion particles
    // const size_t N_particles = static_cast<size_t>(1e9); // Example: 1 billion particles
    const float rs = 1.0f;
    std::cout<<"N_particles: "<<N_particles<<"\n";

    // Initialize random positions in the range [0, 2*rs] using float precision
    std::vector<Particle> positions(N_particles);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, rs * 2.0f);

    for (size_t i = 0; i < N_particles; ++i) {
        positions[i] = {dis(gen), dis(gen), dis(gen)};
    }
    std::cout<<"pos: "<<positions[0].x<<"\n";

    // Parameters for Perlin motion
    const int N_iter = 14;
    const float scale = 0.1f;
    const float step_size = 2.0f;

    // Apply Perlin motion for the given number of iterations
    double start_time, end_time;
    start_time = omp_get_wtime();
    for (int i = 0; i < N_iter; ++i) {
        positions = Perlin_motion(positions, scale, 4, step_size);
    }
    end_time = omp_get_wtime();
    std::cout << "Tree Method completed in " << (end_time - start_time) << " seconds." << std::endl;

    // Output the result
    std::cout << "New positions calculated for " << positions.size() << " particles." << std::endl;
    std::cout<<"pos: "<<positions[0].x<<"\n";

    return 0;
}
