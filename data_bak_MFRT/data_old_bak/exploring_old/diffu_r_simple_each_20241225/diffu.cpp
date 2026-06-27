#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm> // For std::min_element, std::max_element, std::sort
#include <numeric>   // For std::accumulate
#include "diffu.h"

float M_total;
float R0;
float v0;
size_t N_particles;
size_t N_samples;
float b90;
float coef_const_Diffu;

std::vector<float> mean_3 = {0.f, 0.f, 0.f}; //user setting
std::vector<float> dispersion_3 = {60.f, 60.f, 60.f}; //user setting



std::vector<float> get_center(const std::vector<PositionSpace>& pos){
    float xc=0.f, yc=0.f, zc=0.f;
    size_t N = pos.size();
    for(size_t i=0;i<N;++i){
        xc += pos[i].x;
        yc += pos[i].y;
        zc += pos[i].z;
    }
    xc /= N, yc /= N, zc /= N;
    return {xc, yc, zc};
}

float get_mean_radius(const std::vector<PositionSpace>& particles, bool is_consider_center){
    float xc=0.f, yc=0.f, zc=0.f;
    float r_mean=0.f, dx, dy, dz;
    size_t N = particles.size();
    if(is_consider_center){
        for(size_t i=0;i<N;++i){
            xc += particles[i].x;
            yc += particles[i].y;
            zc += particles[i].z;
        }
        xc /= N, yc /= N, zc /= N;
    }
    for(size_t i=0;i<N;++i){
        dx = (particles[i].x-xc);
        dy = (particles[i].y-yc);
        dz = (particles[i].z-zc);
        r_mean += sqrt(dx*dx+dy*dy+dz*dz);
    }
    r_mean /= N;
    return r_mean;
}

float get_mean_radius(const std::vector<VelocitySpace>& particles, bool is_consider_center){
    float xc=0.f, yc=0.f, zc=0.f;
    float r_mean=0.f, dx, dy, dz;
    size_t N = particles.size();
    if(is_consider_center){
        for(size_t i=0;i<N;++i){
            xc += particles[i].x;
            yc += particles[i].y;
            zc += particles[i].z;
        }
        xc /= N, yc /= N, zc /= N;
    }
    for(size_t i=0;i<N;++i){
        dx = (particles[i].x-xc);
        dy = (particles[i].y-yc);
        dz = (particles[i].z-zc);
        r_mean += sqrt(dx*dx+dy*dy+dz*dz);
    }
    r_mean /= N;
    return r_mean;
}

void translate_to_center(std::vector<PositionSpace>& particles){
    size_t N = particles.size();
    auto center = get_center(particles);
    for(size_t i=0;i<N;++i){
        particles[i].x -= center[0];
        particles[i].y -= center[1];
        particles[i].z -= center[2];
    }
}

void translate_to_center(std::vector<VelocitySpace>& particles){
    size_t N = particles.size();
    auto center = get_center(particles);
    for(size_t i=0;i<N;++i){
        particles[i].x -= center[0];
        particles[i].y -= center[1];
        particles[i].z -= center[2];
    }
}

void readjust_positions(std::vector<PositionSpace>& positions, float r_mean){
    size_t N = positions.size();
    auto center = get_center(positions);
    for(size_t i=0;i<N;++i){
        positions[i].x -= center[0];
        positions[i].y -= center[1];
        positions[i].z -= center[2];
    }

    float old_r_mean = get_mean_radius(positions);
    float scaling_factor = r_mean / old_r_mean;
    for(size_t i=0;i<N;++i){
        positions[i].x *= scaling_factor;
        positions[i].y *= scaling_factor;
        positions[i].z *= scaling_factor;
    }
}

std::vector<float> get_center(const std::vector<VelocitySpace>& velocities) {
    std::vector<float> mean = {0.0f, 0.0f, 0.0f};
    size_t n = velocities.size();
    for (const auto& vel : velocities) {
        mean[0] += vel.x;
        mean[1] += vel.y;
        mean[2] += vel.z;
    }
    mean[0] /= n;
    mean[1] /= n;
    mean[2] /= n;
    return mean;
}

float calculate_quadratic_mean(const std::vector<VelocitySpace>& velocities) {
    float sum_of_squares = 0.0f;
    for (const auto& vel : velocities) {
        sum_of_squares += vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    }
    return std::sqrt(sum_of_squares / velocities.size());
}

float calculate_quadratic_mean(const std::vector<PositionSpace>& velocities) {
    float sum_of_squares = 0.0f;
    for (const auto& vel : velocities) {
        sum_of_squares += vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    }
    return std::sqrt(sum_of_squares / velocities.size());
}

void readjust_velocities(std::vector<VelocitySpace>& velocities, const std::vector<float>& mean_3, const std::vector<float>& dispersion_3) {
    // Step 1: Calculate the old mean and subtract it from all velocities
    std::vector<float> old_mean = get_center(velocities);
    for (auto& vel : velocities) {
        vel.x -= old_mean[0];
        vel.y -= old_mean[1];
        vel.z -= old_mean[2];
    }

    // Step 2: Add the new mean to all velocities
    for (auto& vel : velocities) {
        vel.x += mean_3[0];
        vel.y += mean_3[1];
        vel.z += mean_3[2];
    }

    // Step 3: Calculate the old quadratic mean
    float old_quadratic_mean = calculate_quadratic_mean(velocities);

    // Step 4: Calculate the target quadratic mean
    float quadratic_mean = std::sqrt(mean_3[0] * mean_3[0] + mean_3[1] * mean_3[1] + mean_3[2] * mean_3[2] +
                                     dispersion_3[0] * dispersion_3[0] + dispersion_3[1] * dispersion_3[1] +
                                     dispersion_3[2] * dispersion_3[2]);

    // Step 5: Apply the scaling factor
    float scaling_factor = quadratic_mean / old_quadratic_mean;
    for (auto& vel : velocities) {
        vel.x *= scaling_factor;
        vel.y *= scaling_factor;
        vel.z *= scaling_factor;
    }
}

void print_particles_info(const std::vector<PositionSpace>& particles){
    std::cout<<"print_particles_info: \n";
    auto N = particles.size();
    auto mean_radius = get_mean_radius(particles);
    auto qm = calculate_quadratic_mean(particles);
    auto center = get_center(particles);
    print_scalars(N, mean_radius, qm);//, "N, mean_radius, qm");
    print_vectors(center);//, "center");
    return ;
}

void print_all_particles(const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, size_t print_interval){
    size_t N = pos.size();
    size_t interval;
    if(print_interval<1 || print_interval>N){
        std::cerr<<"Wrong value of print_interval, set to 1 and print.\n";
        interval = 1;
    }else{
        interval = print_interval;
    }
    std::cout<<"\nprint_all_particles: \n";
    for(size_t i=0;i<N;i+=interval){
        std::cout<<i<<": "
            <<pos[i].x<<" "<<pos[i].y<<" "<<pos[i].z<<"; "
            <<vel[i].x<<" "<<vel[i].y<<" "<<vel[i].z<<" "
        <<"\n";
    }
    std::cout<<"\n";
    return ;
}

void print_particles_info(const std::vector<VelocitySpace>& particles){
    
    auto N = particles.size();
    auto mean_radius = get_mean_radius(particles);
    auto qm = calculate_quadratic_mean(particles);
    auto center = get_center(particles);
    print_scalars(N, mean_radius, qm);
    print_vectors(center);
    return ;
    return ;
}

void print_galaxy_setting_info(){
    std::cout<<"\ngalaxy info: \n"
    <<"M_total = "<<M_total<<", \n"
    <<"N_samples = "<<N_samples<<", \n"
    <<"R0 = "<<R0<<", \n"
    <<"v0 = "<<v0<<", \n"
    <<"coef_const_Diffu = "<<coef_const_Diffu<<".\n";
    return ;
}

bool readParticles_from_txt(const std::string& file_path, std::vector<PositionSpace>& particles, size_t N_particles) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << file_path << std::endl;
        return false;
    }

    particles.resize(N_particles);
    std::string line;
    size_t idx_p = 0;
    while (idx_p<N_particles) {
        // std::cout<<idx_p<<" ";
        std::getline(file, line);
        std::istringstream sline(line);
        sline >> particles[idx_p].x >> particles[idx_p].y >> particles[idx_p].z;
        idx_p++;
    }

    // std::string line;
    // size_t idx_p = 0;
    // PositionSpace p;
    // while (std::getline(file, line)) {
    //     std::istringstream sline(line);
    //     sline >> p.x >> p.y >> p.z;
    //     particles.push_back(p);
    //     idx_p++;
    // }

    file.close();
    return true;
}

void write_xv_to_binary(const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }

    size_t N_genrate = pos.size();
    for (size_t i = 0; i < N_genrate; ++i) {
        file.write(reinterpret_cast<const char*>(&pos[i]), sizeof(PositionSpace));
        file.write(reinterpret_cast<const char*>(&vel[i]), sizeof(VelocitySpace));
    }

    file.close();
    std::cout << "Samples written to binary file: " << filename << std::endl;
}

bool read_xv_from_binary(const std::string& filename, std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N_particles) {
    // Open the binary file for reading
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for reading." << std::endl;
        return false;
    }

    // Resize the vectors to hold the data
    pos.resize(N_particles);
    vel.resize(N_particles);

    // Read position and velocity data
    for (size_t i = 0; i < N_particles; ++i) {
        file.read(reinterpret_cast<char*>(&pos[i]), sizeof(PositionSpace));
        file.read(reinterpret_cast<char*>(&vel[i]), sizeof(VelocitySpace));
        if (!file) {
            std::cerr << "Error: Unexpected end of file while reading data." << std::endl;
            return false;
        }
    }

    file.close();
    std::cout << "Samples read from binary file, done." << filename << std::endl;
    return true;
}

bool save_Diffur_to_binary(const std::string& filename, const std::vector<float>& D_direct, const std::vector<float>& D_tree){
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return false;
    }

    size_t N_generate = D_tree.size();
    if (D_direct.size() != N_generate) {
        std::cerr << "Error: Size mismatch between D_direct and D_tree." << std::endl;
        return false;
    }

    std::vector<size_t> IDs;
    IDs.resize(N_generate);
    for (size_t i = 0; i < N_generate; ++i) {
        IDs[i] = i+1;
        file.write(reinterpret_cast<const char*>(&IDs[i]), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&D_direct[i]), sizeof(float));
        file.write(reinterpret_cast<const char*>(&D_tree[i]), sizeof(float));
    }

    file.close();
    std::cout << "Diffur written to binary file: " << filename << std::endl;
    return true;
}

bool save_Diffur_to_txt(const std::string& filename, const std::vector<float>& D_direct, const std::vector<float>& D_tree) {
    // Open the file for writing in text mode
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return false;
    }

    // Check that the sizes of D_direct and D_tree match
    size_t N_generate = D_tree.size();
    if (D_direct.size() != N_generate) {
        std::cerr << "Error: Size mismatch between D_direct and D_tree." << std::endl;
        return false;
    }

    // Write the data to the text file
    file << "ID\tD_direct\tD_tree\n"; // Header for better readability
    for (size_t i = 0; i < N_generate; ++i) {
        file << (i + 1) << "\t" << D_direct[i] << "\t" << D_tree[i] << "\n";
    }

    file.close();
    std::cout << "Diffur written to txt file: " << filename << std::endl;
    return true;
}

float calculateDiffusionCoefficient_DS(const std::vector<PositionSpace>& particles, const PositionSpace& target, size_t index_target) {
    float D_i = 0.0f;
    float b90_squared = b90*b90;
    float dx, dy, dz, r_ij_squared;
    for (size_t j = 0; j < N_particles; ++j) {
        if (index_target == j) continue;

        dx = target.x - particles[j].x;
        dy = target.y - particles[j].y;
        dz = target.z - particles[j].z;
        r_ij_squared = dx * dx + dy * dy + dz * dz;

        D_i += r_ij_squared / ((r_ij_squared + b90_squared) * (r_ij_squared + b90_squared));
    }
    return D_i*coef_const_Diffu;
}


float diffusion_reference_value_cylinder(
    float R_minus, bool is_using_formula
){
    float m = M_total/N_particles;
    float R_soften;
    if(R_minus>0.){
        R_soften = R_minus;
    }
    else{
        R_soften = R0/(lambda2*N_particles);
    }
    float log_Alpha = std::log(R0/R_soften);
    float n0 = N_particles/(M_PI*pow(R0*1.5, 2)*2.*R0); //roughly cylinder region
    float IRp2 = 8./9.*N_particles*log_Alpha;
    float coef_const = 27.*M_PI/2.*pow(G*M_total, 2)*n0/(v0*pow(N_particles, 3));
    float Diffu_0;
    if(is_using_formula){
        Diffu_0 = 8.*M_PI*pow(G*M_total, 2)*n0*log_Alpha/(v0*pow(N_particles, 2));
    }
    else{
        Diffu_0 = coef_const * IRp2;
    }
    return Diffu_0;
}

void directSummation(const std::vector<PositionSpace>& particles, std::vector<float>& D_direct, size_t N_particles) {
    float b90_squared = b90*b90;

    for (size_t i = 0; i < N_particles; ++i) {
        float D_i = 0.0f;
        float dx, dy, dz, r_ij_squared;
        for (size_t j = 0; j < N_particles; ++j) {
            if (i == j) continue;

            dx = particles[i].x - particles[j].x;
            dy = particles[i].y - particles[j].y;
            dz = particles[i].z - particles[j].z;
            r_ij_squared = dx * dx + dy * dy + dz * dz;

            D_i += r_ij_squared / ((r_ij_squared + b90_squared) * (r_ij_squared + b90_squared));
        }
        D_direct[i] = D_i*coef_const_Diffu;
    }
}

Diffu_info calculate_diffu_statistics(const std::vector<float>& data, std::string file_statistics, bool is_save){
    Diffu_info Diffu_statistics;
    if (data.empty()) {
        std::cout<<"The vector is empty. Cannot calculate statistics. Exit.\n";
        exit(0);
    }

    float min_value = *std::min_element(data.begin(), data.end());
    float max_value = *std::max_element(data.begin(), data.end());
    
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    float mean_value = sum / data.size();
    
    std::vector<float> sorted_data = data; // Create a copy to sort
    std::sort(sorted_data.begin(), sorted_data.end());
    float median_value;
    size_t size = sorted_data.size();
    if (size % 2 == 0) {
        // Even number of elements: median is the average of the two middle values
        median_value = (sorted_data[size / 2 - 1] + sorted_data[size / 2]) / 2.0f;
    } else {
        // Odd number of elements: median is the middle value
        median_value = sorted_data[size / 2];
    }

    Diffu_statistics.N_particles = size;
    Diffu_statistics.Diffu_0 = diffusion_reference_value_cylinder();
    Diffu_statistics.Diffu_mean = mean_value;
    Diffu_statistics.Diffu_median = median_value;
    Diffu_statistics.Diffu_min = min_value;
    Diffu_statistics.Diffu_max = max_value;

    std::ofstream txt_file_out(file_statistics);
    txt_file_out<<Diffu_statistics.N_particles<<"\t"<<Diffu_statistics.Diffu_0<<"\t"
        <<Diffu_statistics.Diffu_mean<<"\t"<<Diffu_statistics.Diffu_median<<" ";
    txt_file_out.close();

    return Diffu_statistics;
}

void print_diffu_info(const Diffu_info& Diffu_statistics){
    std::cout<<"\nDiffu info: \n"
        <<"N_particles = "<<Diffu_statistics.N_particles<<", \n"
        <<"Diffu_0 = "<<Diffu_statistics.Diffu_0<<", \n"
        <<"Diffu_mean = "<<Diffu_statistics.Diffu_mean<<", \n"
        <<"Diffu_median = "<<Diffu_statistics.Diffu_median<<", \n"
        // <<"Diffu_min = "<<Diffu_statistics.Diffu_min<<", \n"
        // <<"Diffu_max = "<<Diffu_statistics.Diffu_max<<", \n"
    ;
    return ;
}
