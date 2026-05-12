#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm> // For std::min_element, std::max_element, std::sort
#include <numeric>   // For std::accumulate
#include "diffu.h"

PrecisionSetting M_total;
size_t N_particles;
size_t N_samples;

PrecisionSetting R0;
PrecisionSetting Rm;
PrecisionSetting b90;
PrecisionSetting R_epsilon;
PrecisionSetting coef_const_Diffu = -1.;

PrecisionSetting v0 = -1.;
PrecisionSetting v2m = -1.;
PrecisionSetting v2m_sqrt = -1.;
PrecisionSetting v_epsilon = -1.;
PrecisionSetting coef_const_Diffu_vel = -1.;

PrecisionSetting n0 = -1.;
PrecisionSetting ln_Lambda = -1.;
PrecisionSetting GXX = 0.15;
PrecisionSetting ratio_Diffu_old = -1.;
PrecisionSetting ratio_Diffu_old_vel = -1.;
PrecisionSetting Diffu_0_unit = -1.;
PrecisionSetting Diffu_ref = -1.;
PrecisionSetting coef_Diffu_parallel_iso = -1.;
PrecisionSetting coef_Diffu_tensor_uniform = -1.;
PrecisionSetting coef_Diffu_separable = -1.;

std::vector<PrecisionSetting> vmean_3; //user setting
std::vector<PrecisionSetting> dispersion_3_iso_ratio;
std::vector<PrecisionSetting> dispersion_3_high_ratio; //user setting

std::vector<PrecisionSetting> qpers_display = {
    0.0,    0.0001, 0.0005, 0.001,  0.005,  0.01,  0.05, 
    0.1,    0.15,   0.2,    0.25,   0.3,    0.35, 
    0.4,    0.45,   0.5,    0.55,   0.6,    0.65, 
    0.7,    0.75,   0.8,    0.85,   0.9,    0.95, 
    0.99,   0.995,  0.999,  0.9995, 0.9999, 1.0
};



std::vector<PrecisionSetting> get_center(const std::vector<PositionSpace>& pos){
    PrecisionSetting xc=0.f, yc=0.f, zc=0.f;
    size_t N = pos.size();
    for(size_t i=0;i<N;++i){
        xc += pos[i].x;
        yc += pos[i].y;
        zc += pos[i].z;
    }
    xc /= N, yc /= N, zc /= N;
    return {xc, yc, zc};
}

PrecisionSetting get_mean_radius(const std::vector<PositionSpace>& particles, bool is_consider_center){
    PrecisionSetting xc=0.f, yc=0.f, zc=0.f;
    PrecisionSetting r_mean=0.f, dx, dy, dz;
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
        r_mean += std::sqrt(dx*dx+dy*dy+dz*dz);
    }
    r_mean /= N;
    return r_mean;
}

PrecisionSetting get_mean_radius(const std::vector<VelocitySpace>& particles, bool is_consider_center){
    PrecisionSetting xc=0.f, yc=0.f, zc=0.f;
    PrecisionSetting r_mean=0.f, dx, dy, dz;
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
        r_mean += std::sqrt(dx*dx+dy*dy+dz*dz);
    }
    r_mean /= N;
    return r_mean;
}

PrecisionSetting calculate_v0_vel(const std::vector<VelocitySpace>& particles) {
    // return get_mean_radius(particles, true);
    return calculate_quadratic_mean(particles);
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

void readjust_positions(std::vector<PositionSpace>& positions, PrecisionSetting r_mean){
    size_t N = positions.size();
    auto center = get_center(positions);
    for(size_t i=0;i<N;++i){
        positions[i].x -= center[0];
        positions[i].y -= center[1];
        positions[i].z -= center[2];
    }

    PrecisionSetting old_r_mean = get_mean_radius(positions);
    PrecisionSetting scaling_factor = r_mean / old_r_mean;
    for(size_t i=0;i<N;++i){
        positions[i].x *= scaling_factor;
        positions[i].y *= scaling_factor;
        positions[i].z *= scaling_factor;
    }
}

std::vector<PrecisionSetting> get_center(const std::vector<VelocitySpace>& velocities) {
    std::vector<PrecisionSetting> mean = {0.0f, 0.0f, 0.0f};
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

PrecisionSetting calculate_quadratic_mean(const std::vector<VelocitySpace>& velocities) {
    PrecisionSetting sum_of_squares = 0.0f;
    for (const auto& vel : velocities) {
        sum_of_squares += vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    }
    return std::sqrt(sum_of_squares / velocities.size());
}

PrecisionSetting calculate_quadratic_mean(const std::vector<PositionSpace>& velocities) {
    PrecisionSetting sum_of_squares = 0.0f;
    for (const auto& vel : velocities) {
        sum_of_squares += vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
    }
    return std::sqrt(sum_of_squares / velocities.size());
}

PrecisionSetting calculate_speed_square_mean(const std::vector<PrecisionSetting>& vm, const std::vector<PrecisionSetting>& dispersion_diag){
    PrecisionSetting v2m = 0.;
    for(int i=0;i<3;++i){
        v2m += vm[i]*vm[i] + dispersion_diag[i]*dispersion_diag[i];
    }
    return v2m;
}

PrecisionSetting calculate_dispersion_iso(PrecisionSetting v2m_sqrt, const std::vector<PrecisionSetting>& vm){
    PrecisionSetting mean_v2 = vm[0]*vm[0]+vm[1]*vm[1]+vm[2]*vm[2];
    return std::sqrt((v2m_sqrt*v2m_sqrt-mean_v2)/3.);
}

PrecisionSetting norm_vector(const std::vector<PrecisionSetting>& vec) {
    if (vec.size() != 3)
        throw std::runtime_error("Input vector must have 3 elements.");
    PrecisionSetting sum = 0.0;
    for (const auto& x : vec) {
        sum += x * x;
    }
    return std::sqrt(sum);
}

/*  Compute the isotropic dispersion given a target RMS speed (v2m_sqrt)
    and a fixed 3-vector for the mean velocity (vmean_3). 
    vmean_3 is assumed to have three components.
*/
std::vector<PrecisionSetting> rescale_dispersion_keep_ratio(
    const std::vector<PrecisionSetting>& vmean_three_fixed, const std::vector<PrecisionSetting>& sigma_three_old, 
    PrecisionSetting v2m_sqrt_target
){
    if (vmean_three_fixed.size() != 3 || sigma_three_old.size() != 3) {
        throw std::runtime_error("Both input vectors must have 3 elements.");
    }
    
    // Compute the total dispersion of the old vector.
    PrecisionSetting sigma_total_old = norm_vector(sigma_three_old);
    if (sigma_total_old == 0.0) {
        throw std::runtime_error("sigma_three_old has zero norm; cannot rescale.");
    }
    
    // Compute the ratio for each component.
    std::vector<PrecisionSetting> ratio(3);
    for (size_t i = 0; i < 3; ++i) {
        ratio[i] = sigma_three_old[i] / sigma_total_old;
    }
    
    // Compute the total target dispersion: 
    // sigma_total_target = sqrt(v2m_sqrt_target^2 - ||vmean_three_fixed||^2)
    PrecisionSetting mean_vnorm = norm_vector(vmean_three_fixed);
    PrecisionSetting sigma_total_target = std::sqrt(v2m_sqrt_target * v2m_sqrt_target - mean_vnorm * mean_vnorm);
    
    // Rescale the dispersion for each component.
    std::vector<PrecisionSetting> sigma_three_target(3);
    for (size_t i = 0; i < 3; ++i) {
        sigma_three_target[i] = sigma_total_target * ratio[i];
    }
    
    return sigma_three_target;
}

void readjust_velocities_v2m_sqrt(
    std::vector<VelocitySpace>& velocities, const std::vector<PrecisionSetting>& vmean_3, const std::vector<PrecisionSetting>& dispersion_3
){
    // Step 1: Calculate the old mean and subtract it from all velocities
    std::vector<PrecisionSetting> old_mean = get_center(velocities);
    for (auto& vel : velocities) {
        vel.x -= old_mean[0];
        vel.y -= old_mean[1];
        vel.z -= old_mean[2];
    }

    // Step 2: Add the new mean to all velocities
    for (auto& vel : velocities) {
        vel.x += vmean_3[0];
        vel.y += vmean_3[1];
        vel.z += vmean_3[2];
    }

    // Step 3: Calculate the old quadratic mean
    PrecisionSetting old_quadratic_mean = calculate_quadratic_mean(velocities);

    // Step 4: Calculate the target quadratic mean
    PrecisionSetting quadratic_mean = std::sqrt(
        vmean_3[0] * vmean_3[0] + vmean_3[1] * vmean_3[1] + vmean_3[2] * vmean_3[2] +dispersion_3[0] * dispersion_3[0] 
        + dispersion_3[1] * dispersion_3[1] + dispersion_3[2] * dispersion_3[2]
    );

    // Step 5: Apply the scaling factor
    PrecisionSetting scaling_factor = quadratic_mean / old_quadratic_mean; //?? three component
    for (auto& vel : velocities) {
        vel.x *= scaling_factor;
        vel.y *= scaling_factor;
        vel.z *= scaling_factor;
    }
}

void copy_xv_sample(const std::vector<PositionSpace>& p_ori, std::vector<PositionSpace>& p_another){
    size_t N = p_ori.size();
    p_another.resize(N);
    for(int i=0;i<N;++i){
        p_another[i].x = p_ori[i].x;
        p_another[i].y = p_ori[i].y;
        p_another[i].z = p_ori[i].z;
    }
}

void copy_xv_sample(const std::vector<VelocitySpace>& p_ori, std::vector<VelocitySpace>& p_another){
    size_t N = p_ori.size();
    p_another.resize(N);
    for(int i=0;i<N;++i){
        p_another[i].x = p_ori[i].x;
        p_another[i].y = p_ori[i].y;
        p_another[i].z = p_ori[i].z;
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

void print_galaxy_setting_info(std::string descrp){
    std::cout<<"\ngalaxy info for "<<descrp<<": \n"
    <<"M_total = "<<M_total<<", \n"
    <<"N_particles = "<<N_particles<<"; \n"
    <<"Rm = "<<Rm<<", \n"
    <<"R_epsilon = "<<R_epsilon<<", \n"
    <<"n0 = "<<n0<<", \n"
    <<"ln_Lambda = "<<ln_Lambda<<", \n"
    <<"coef_const_Diffu = "<<coef_const_Diffu<<"; \n"
    <<"v0 = "<<v0<<", \n"
    <<"v2m_sqrt = "<<v2m_sqrt<<", \n"
    <<"v_epsilon = "<<v_epsilon<<", \n"
    <<"coef_const_Diffu_vel = "<<coef_const_Diffu_vel<<"; \n"
    <<"ratio_Diffu_old = "<<ratio_Diffu_old<<", \n"
    <<"ratio_Diffu_old_vel = "<<ratio_Diffu_old_vel<<", \n"
    <<"Diffu_0_unit = "<<Diffu_0_unit<<", \n"
    <<"Diffu_ref = "<<Diffu_ref<<".\n";
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

bool load_xv_from_txt(std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, const std::string& filename){
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    pos.clear();
    vel.clear();

    std::string line;
    size_t line_number = 0;
    while (std::getline(infile, line)) {
        ++line_number;
        std::istringstream iss(line);
        PrecisionSetting x, y, z, vx, vy, vz;
        if (!(iss >> x >> y >> z >> vx >> vy >> vz)) {
            std::cerr << "Warning: Line " << line_number
                      << " is malformed and will be skipped." << std::endl;
            continue;
        }
        pos.push_back(PositionSpace{x, y, z}); //note: ctrl+shift+P to c_cpp_properties, then change to {"cppStandard": "c++17",}
        vel.push_back(VelocitySpace{vx, vy, vz});
    }

    return true;
}

void write_xv_to_binary(const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }

    size_t N_generate = pos.size();
    for (size_t i = 0; i < N_generate; ++i) {
        file.write(reinterpret_cast<const char*>(&pos[i]), sizeof(PositionSpace));
        file.write(reinterpret_cast<const char*>(&vel[i]), sizeof(VelocitySpace));
    }

    file.close();
    std::cout << "Samples written to binary file: " << filename << std::endl;
}

void write_xv_to_txt(const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return;
    }
    size_t N_generate = pos.size();

    for (size_t i = 0; i < N_generate; ++i) {
        file<<pos[i].x<<"\t"<<pos[i].y<<"\t"<<pos[i].z<<"\t"
        <<vel[i].x<<"\t"<<vel[i].y<<"\t"<<vel[i].z<<"\n";
    }

    file.close();
    std::cout << "Samples written to txt file: " << filename << std::endl;
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

bool save_Diffur_to_binary(const std::string& filename, const std::vector<PrecisionSetting>& D_direct, const std::vector<PrecisionSetting>& D_tree){
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
        file.write(reinterpret_cast<const char*>(&D_direct[i]), sizeof(PrecisionSetting));
        file.write(reinterpret_cast<const char*>(&D_tree[i]), sizeof(PrecisionSetting));
    }

    file.close();
    std::cout << "Diffur written to binary file: " << filename << std::endl;
    return true;
}

bool read_Diffur_from_binary(const std::string& filename, std::vector<PrecisionSetting>& D_direct, std::vector<PrecisionSetting>& D_tree) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for reading." << std::endl;
        return false;
    }

    // Clear vectors to store data
    D_direct.clear();
    D_tree.clear();

    size_t id; // Placeholder for IDs
    PrecisionSetting direct_value, tree_value;

    // Read the binary file data
    while (file.read(reinterpret_cast<char*>(&id), sizeof(size_t))) {
        if (!file.read(reinterpret_cast<char*>(&direct_value), sizeof(PrecisionSetting))) break;
        if (!file.read(reinterpret_cast<char*>(&tree_value), sizeof(PrecisionSetting))) break;

        // Store values in corresponding vectors
        D_direct.push_back(direct_value);
        D_tree.push_back(tree_value);
    }

    file.close();
    std::cout << "Diffur read from binary file: " << filename << std::endl;
    return true;
}

bool save_Diffur_to_txt(const std::string& filename, const std::vector<PrecisionSetting>& D_direct, const std::vector<PrecisionSetting>& D_tree) {
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
    file << "#ID\tD_direct\tD_tree\n"; // Header for better readability
    for (size_t i = 0; i < N_generate; ++i) {
        file << (i + 1) << "\t" << D_direct[i] << "\t" << D_tree[i] << "\n";
    }

    file.close();
    std::cout << "Diffur written to txt file: " << filename << std::endl;
    return true;
}

bool save_percentile_to_txt(const std::string& filename, const std::vector<PrecisionSetting>& q, const std::vector<PrecisionSetting>& a) {
    // Open the file for writing in text mode
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        return false;
    }

    // Check that the sizes of D_direct and D_tree match
    size_t N = q.size();
    if (q.size() != N) {
        std::cerr << "Error: Size mismatch." << std::endl;
        return false;
    }

    // Write the data to the text file
    file << "#q\ta\n"; // Header for better readability
    for (size_t i = 0; i < N; ++i) {
        file << q[i] << "\t" << a[i] << "\n";
    }

    file.close();
    std::cout << "Saved the percentile to: " << filename << std::endl;
    return true;
}

bool save_inside_counts(
    const std::string& filename,
    const std::vector<PrecisionSetting>& radii,
    const std::vector<PrecisionSetting>& inscs
) {
    if (radii.size() != inscs.size()) {
        std::cerr << "[Error] Radii and inside_counts vectors have different sizes.\n";
        return false;
    }

    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "[Error] Failed to open file for writing: " << filename << "\n";
        return false;
    }

    fout << std::setprecision(12);  // high precision
    for (size_t i = 0; i < radii.size(); ++i) {
        fout << radii[i] << " " << inscs[i] << "\n";
    }

    fout.close();
    std::cout << "Saved inside_counts to file: " << filename << "\n";
    return true;
}

// Save all diffusion tensors to a binary file.
// The file begins with a size_t indicating the number of entries,
// followed by, for each particle, the following data (in order):
// ID (size_t), D_eff (PrecisionSetting), D_11, D_22, D_33, D_12, D_13, D_23 (all PrecisionSetting).
bool save_all_diffu_tensors_to_binary(const std::string &filename, const std::vector<Diffu_tensor_vel> &diffu) {
    std::ofstream ofs(filename.c_str(), std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[Error] Could not open file " << filename << " for writing.\n";
        return false;
    }
    
    size_t N = diffu.size();
    ofs.write(reinterpret_cast<const char *>(&N), sizeof(size_t));
    
    for (size_t i = 0; i < N; ++i) {
        size_t id = i + 1;
        // Calculate D_eff = (D[0][0]+D[1][1]+D[2][2])/3.0
        PrecisionSetting D_eff = (diffu[i].D[0][0] + diffu[i].D[1][1] + diffu[i].D[2][2]) / 3.0;
        
        ofs.write(reinterpret_cast<const char *>(&id), sizeof(size_t));
        ofs.write(reinterpret_cast<const char *>(&D_eff), sizeof(PrecisionSetting));
        // Save the 6 independent components: D[0][0], D[1][1], D[2][2], D[0][1], D[0][2], D[1][2]
        ofs.write(reinterpret_cast<const char *>(&(diffu[i].D[0][0])), sizeof(PrecisionSetting));
        ofs.write(reinterpret_cast<const char *>(&(diffu[i].D[1][1])), sizeof(PrecisionSetting));
        ofs.write(reinterpret_cast<const char *>(&(diffu[i].D[2][2])), sizeof(PrecisionSetting));
        ofs.write(reinterpret_cast<const char *>(&(diffu[i].D[0][1])), sizeof(PrecisionSetting));
        ofs.write(reinterpret_cast<const char *>(&(diffu[i].D[0][2])), sizeof(PrecisionSetting));
        ofs.write(reinterpret_cast<const char *>(&(diffu[i].D[1][2])), sizeof(PrecisionSetting));
    }
    
    ofs.close();
    std::cout << "Saved " << N << " diffusion tensors to binary file: " << filename << "\n";
    return true;
}

// Save all diffusion tensors to a text file.
// Each line of the text file will contain:
// ID D_eff D_11 D_22 D_33 D_12 D_13 D_23
// separated by spaces.
bool save_all_diffu_tensors_to_txt(const std::string &filename, const std::vector<Diffu_tensor_vel> &diffu) {
    std::ofstream ofs(filename.c_str());
    if (!ofs.is_open()) {
        std::cerr << "[Error] Could not open file " << filename << " for writing.\n";
        return false;
    }
    
    ofs << std::setprecision(12) << std::fixed;
    size_t N = diffu.size();
    for (size_t i = 0; i < N; ++i) {
        size_t id = i + 1;
        PrecisionSetting D_eff = (diffu[i].D[0][0] + diffu[i].D[1][1] + diffu[i].D[2][2]) / 3.0;
        ofs << id << " " << D_eff << " " 
            << diffu[i].D[0][0] << " " 
            << diffu[i].D[1][1] << " " 
            << diffu[i].D[2][2] << " " 
            << diffu[i].D[0][1] << " " 
            << diffu[i].D[0][2] << " " 
            << diffu[i].D[1][2] << "\n";
    }
    
    ofs.close();
    std::cout << "Saved " << N << " diffusion tensors to text file: " << filename << "\n";
    return true;
}

// Read diffusion tensors from a binary file saved by the above function.
// The file is assumed to start with a size_t number indicating the number of records,
// and each record consists of: ID (size_t), D_eff (PrecisionSetting), D_11, D_22, D_33, D_12, D_13, D_23.
// We reconstruct the full 3x3 symmetric tensor assuming:
//    D[1][0] = D[0][1], D[2][0] = D[0][2], and D[2][1] = D[1][2].
bool read_diffu_tensors_from_binary(const std::string &filename, std::vector<Diffu_tensor_vel> &diffu) {
    std::ifstream ifs(filename.c_str(), std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "[Error] Could not open file " << filename << " for reading.\n";
        return false;
    }
    
    size_t N = 0;
    ifs.read(reinterpret_cast<char *>(&N), sizeof(size_t));
    diffu.resize(N);
    
    for (size_t i = 0; i < N; ++i) {
        size_t id = 0;
        PrecisionSetting D_eff = 0.0;
        PrecisionSetting D11, D22, D33, D12, D13, D23;
        
        ifs.read(reinterpret_cast<char *>(&id), sizeof(size_t));
        ifs.read(reinterpret_cast<char *>(&D_eff), sizeof(PrecisionSetting));
        ifs.read(reinterpret_cast<char *>(&D11), sizeof(PrecisionSetting));
        ifs.read(reinterpret_cast<char *>(&D22), sizeof(PrecisionSetting));
        ifs.read(reinterpret_cast<char *>(&D33), sizeof(PrecisionSetting));
        ifs.read(reinterpret_cast<char *>(&D12), sizeof(PrecisionSetting));
        ifs.read(reinterpret_cast<char *>(&D13), sizeof(PrecisionSetting));
        ifs.read(reinterpret_cast<char *>(&D23), sizeof(PrecisionSetting));
        
        // Reconstruct full symmetric tensor
        diffu[i].D[0][0] = D11;
        diffu[i].D[1][1] = D22;
        diffu[i].D[2][2] = D33;
        diffu[i].D[0][1] = D12;  diffu[i].D[1][0] = D12;
        diffu[i].D[0][2] = D13;  diffu[i].D[2][0] = D13;
        diffu[i].D[1][2] = D23;  diffu[i].D[2][1] = D23;
    }
    
    ifs.close();
    std::cout << "Loaded " << N << " diffusion tensors from binary file: " << filename << "\n";
    return true;
}

bool save_diffu_statistics(const std::string& filename, const Statistics_info& sta) {
    std::ofstream fout(filename.c_str());
    if (!fout.is_open()) {
        std::cerr << "[Error] Could not open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Use fixed precision for doubles
    fout << std::fixed << std::setprecision(12);
    
    // Write a header row (optional)
    fout << "#Each line: D_eff D_11 D_22 D_33 D_12 D_13 D_23 other_info."
        "Raws of the 0~6 lines  : Mean Median Min Max"
        "Raws of the 7th line   : N_particles Diffu0 0.0 0.0.\n";
    
    // Line 0: D_eff (effective diffusion coefficient: (1/3)Tr(D))
    fout << sta.Diffu_mean << " "
         << sta.Diffu_median << " "
         << sta.Diffu_min << " "
         << sta.Diffu_max << " \n";
         
    // Line 1: D_11: using the (0,0) component
    fout << sta.Diffu_vel_mean.D[0][0] << " "
         << sta.Diffu_vel_median.D[0][0] << " "
         << sta.Diffu_vel_min.D[0][0] << " "
         << sta.Diffu_vel_max.D[0][0] << " \n";

    // Line 2: D_22: (1,1) component
    fout << sta.Diffu_vel_mean.D[1][1] << " "
         << sta.Diffu_vel_median.D[1][1] << " "
         << sta.Diffu_vel_min.D[1][1] << " "
         << sta.Diffu_vel_max.D[1][1] << " \n";

    // Line 3: D_33: (2,2) component
    fout << sta.Diffu_vel_mean.D[2][2] << " "
         << sta.Diffu_vel_median.D[2][2] << " "
         << sta.Diffu_vel_min.D[2][2] << " "
         << sta.Diffu_vel_max.D[2][2] << " \n";

    // Line 4: D_12: (0,1) component
    fout << sta.Diffu_vel_mean.D[0][1] << " "
         << sta.Diffu_vel_median.D[0][1] << " "
         << sta.Diffu_vel_min.D[0][1] << " "
         << sta.Diffu_vel_max.D[0][1] << " \n";

    // Line 5: D_13: (0,2) component
    fout << sta.Diffu_vel_mean.D[0][2] << " "
         << sta.Diffu_vel_median.D[0][2] << " "
         << sta.Diffu_vel_min.D[0][2] << " "
         << sta.Diffu_vel_max.D[0][2] << " \n";

    // Line 6: D_23: (1,2) component
    fout << sta.Diffu_vel_mean.D[1][2] << " "
         << sta.Diffu_vel_median.D[1][2] << " "
         << sta.Diffu_vel_min.D[1][2] << " "
         << sta.Diffu_vel_max.D[1][2] << " \n";

    // Line 7: other_info
    fout << sta.N_particles << " "
         << sta.Diffu_ref << " "
         << 0.0 << " "
         << 0.0 << " \n";

    fout.close();
    std::cout << "Saved diffusion statistics to: " << filename << std::endl;
    return true;
}

// Function to calculate percentiles
std::vector<PrecisionSetting> np_percentile(const std::vector<PrecisionSetting>& a, const std::vector<PrecisionSetting>& q) {
    // Ensure the input data is not empty
    if (a.empty()) {
        throw std::invalid_argument("Input array 'a' must not be empty.");
    }

    // Ensure the percentiles are within the valid range [0, 1]
    for (const auto& qi : q) {
        if (qi < 0.0 || qi > 1.0) {
            throw std::invalid_argument("Percentile values in 'q' must be in the range [0, 1].");
        }
    }

    // Copy and sort the input array 'a'
    std::vector<PrecisionSetting> sorted_a = a;
    std::sort(sorted_a.begin(), sorted_a.end());

    // Compute the percentiles
    std::vector<PrecisionSetting> percentiles;
    size_t n = sorted_a.size();

    for (const auto& qi : q) {
        // Calculate the index for the percentile
        double index = (n - 1) * qi; // Position in the sorted array
        size_t lower = static_cast<size_t>(std::floor(index)); // Lower bound
        size_t upper = static_cast<size_t>(std::ceil(index));  // Upper bound

        // Interpolate if the index is not an integer
        if (lower == upper) {
            percentiles.push_back(sorted_a[lower]);
        } else {
            double weight = index - lower;
            percentiles.push_back((1.0 - weight) * sorted_a[lower] + weight * sorted_a[upper]);
        }
    }

    return percentiles;
}

PrecisionSetting diffusion_reference_value_cylinder(
    PrecisionSetting R_minus, bool is_using_formula
){
    PrecisionSetting m = M_total/N_particles;
    PrecisionSetting R_soften;
    if(R_minus>0.){
        R_soften = R_minus;
    }
    else{
        R_soften = R0/(lambda4*N_particles);
    }
    PrecisionSetting log_Alpha = std::log(R0/R_soften);
    PrecisionSetting n0 = N_particles/(M_PI*pow(R0*1.5, 2)*2.*R0); //roughly cylinder region
    PrecisionSetting IRp2 = 8./9.*N_particles*log_Alpha;
    PrecisionSetting coef_const = 27.*M_PI/2.*pow(G*M_total, 2)*n0/(v0*pow(N_particles, 3));
    PrecisionSetting Diffu_ref;
    if(is_using_formula){
        Diffu_ref = 8.*M_PI*pow(G*M_total, 2)*n0*log_Alpha/(v0*pow(N_particles, 2));
    }
    else{
        Diffu_ref = coef_const * IRp2;
    }
    return Diffu_ref;
}

// A simple helper struct for 1D statistics.
struct Stats {
    PrecisionSetting mean;
    PrecisionSetting median;
    PrecisionSetting min;
    PrecisionSetting max;
};

/*  Compute statistics (mean, median, min, max) for a vector of numbers. 
    The static keyword here is an internal linkage.
*/
// static Stats computeStats(const std::vector<PrecisionSetting>& vec, bool is_use_abs=true) {
Stats computeStats(const std::vector<PrecisionSetting>& vec, bool is_use_abs=true) {
    if (vec.empty()) {
        throw std::runtime_error("Cannot compute statistics on an empty vector.");
    }
    // Create a new vector to hold the absolute values
    std::vector<PrecisionSetting> absVec(vec.size());
    if(is_use_abs){
        std::transform(vec.begin(), vec.end(), absVec.begin(),
                    [](PrecisionSetting x) { return std::abs(x); });
    }else{
        std::transform(vec.begin(), vec.end(), absVec.begin(),
                    [](PrecisionSetting x) { return x; });
    }
    
    Stats s;
    s.min = *std::min_element(absVec.begin(), absVec.end());
    s.max = *std::max_element(absVec.begin(), absVec.end());
    s.mean = std::accumulate(absVec.begin(), absVec.end(), 0.0) / absVec.size();

    std::vector<PrecisionSetting> sorted = absVec; // copy the absolute values
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0) {
        s.median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    } else {
        s.median = sorted[n / 2];
    }
    return s;
}

Diffu_info calculate_diffu_statistics(const std::vector<PrecisionSetting>& data, std::string file_statistics, bool is_save){
    Diffu_info Diffu_statistics;
    if (data.empty()) {
        std::cout<<"The vector is empty. Cannot calculate statistics. Exit.\n";
        exit(0);
    }

    PrecisionSetting min_value = *std::min_element(data.begin(), data.end());
    PrecisionSetting max_value = *std::max_element(data.begin(), data.end());
    
    PrecisionSetting sum = std::accumulate(data.begin(), data.end(), 0.0f);
    PrecisionSetting mean_value = sum / data.size();
    
    std::vector<PrecisionSetting> sorted_data = data; // Create a copy to sort
    std::sort(sorted_data.begin(), sorted_data.end());
    PrecisionSetting median_value;
    size_t size = sorted_data.size();
    if (size % 2 == 0) {
        // Even number of elements: median is the average of the two middle values
        median_value = (sorted_data[size / 2 - 1] + sorted_data[size / 2]) / 2.0f;
    } else {
        // Odd number of elements: median is the middle value
        median_value = sorted_data[size / 2];
    }

    Diffu_statistics.N_particles = size;
    Diffu_statistics.Diffu_ref = diffusion_reference_value_Diffu_ref();
    Diffu_statistics.Diffu_mean = mean_value;
    Diffu_statistics.Diffu_median = median_value;
    Diffu_statistics.Diffu_min = min_value;
    Diffu_statistics.Diffu_max = max_value;
    Diffu_statistics.Diffu_0_big = Diffu_statistics.Diffu_ref*N_particles/std::log(lambda4*N_particles); //0.9995~0.9998

    std::ofstream txt_file_out(file_statistics);
    txt_file_out<<Diffu_statistics.N_particles<<"\t"<<Diffu_statistics.Diffu_ref<<"\t"
        <<Diffu_statistics.Diffu_mean<<"\t"<<Diffu_statistics.Diffu_median<<" ";
    txt_file_out.close();

    return Diffu_statistics;
}

Statistics_info calculate_diffu_statistics(
    const std::vector<Diffu_tensor_vel>& diffu, std::vector<PrecisionSetting>& diffu_eff
){
    Statistics_info stats;
    size_t N = diffu.size();
    stats.N_particles = N;
    if (N == 0) {
        throw std::runtime_error("No diffusion tensor data available.");
    }
    stats.Diffu_ref = diffusion_reference_value_Diffu_ref();
    stats.Diffu_0_big = -1.;

    // Prepare vector for scalar effective diffusion coefficient for each particle: D_eff = 1/3 * (D[0][0] + D[1][1] + D[2][2])
    ;
    diffu_eff.resize(N);

    // Prepare containers for each tensor component.
    // We use an array of 3x3 vectors, where comp[i][j] holds the value of that component for each particle.
    std::vector<PrecisionSetting> comp[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            comp[i][j].resize(N);

    // Loop over each diffusion tensor and collect statistics.
    for (size_t k = 0; k < N; ++k) {
        const Diffu_tensor_vel& D = diffu[k];
        // Compute the trace and then the effective (per coordinate) diffusion coefficient:
        PrecisionSetting trace = D.D[0][0] + D.D[1][1] + D.D[2][2];
        diffu_eff[k] = trace / 3.0;

        // Save each tensor component.
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                comp[i][j][k] = D.D[i][j];
    }

    // Compute statistics for the scalar effective diffusion coefficient.
    Stats scalarStats = computeStats(diffu_eff);
    stats.Diffu_mean = scalarStats.mean;
    stats.Diffu_median = scalarStats.median;
    stats.Diffu_min = scalarStats.min;
    stats.Diffu_max = scalarStats.max;

    // For each tensor component, compute statistics and store in the corresponding fields.
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Stats compStats = computeStats(comp[i][j]);
            stats.Diffu_vel_mean.D[i][j] = compStats.mean;
            stats.Diffu_vel_median.D[i][j] = compStats.median;
            stats.Diffu_vel_min.D[i][j] = compStats.min;
            stats.Diffu_vel_max.D[i][j] = compStats.max;
        }
    }

    return stats;
}

void print_Diffu_tensor_vel(const Diffu_tensor_vel& tensor){
    std::cout<<"Diffu_tensor = \n";
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 3; ++j){
            std::cout<<tensor.D[i][j]<<" ";
        }
        std::cout<<"\n";
    }
}

void print_diffu_info(const Diffu_info& Diffu_statistics){
    std::cout<<"\nDiffu info: \n"
        <<"N_particles = "<<Diffu_statistics.N_particles<<", \n"
        <<"Diffu_ref = "<<Diffu_statistics.Diffu_ref<<", \n"
        <<"Diffu_mean = "<<Diffu_statistics.Diffu_mean<<", \n"
        <<"Diffu_median = "<<Diffu_statistics.Diffu_median<<", \n"
    ;
    return ;
}

PrecisionSetting diffusion_reference_value_Diffu_ref(){
    // return std::sqrt(6./M_PI)*GXX *ln_Lambda * Diffu_0_unit;
    return 3.0*std::sqrt(6./M_PI)*GXX *ln_Lambda * Diffu_0_unit;
}

void calculate_fixed_const_coefs(){
    // N_particles = N_setting;
    // M_total = M_total_gal*frac_mass;
    // R0 = R0_setting;

    N_samples = N_particles;
    Rm = R0/lambda2;
    R_epsilon = R0 / (lambda4 * N_particles);
    b90 = R_epsilon;
    
    v0 = calculate_v0(); //set the typical cross speed as the virial speed
    v2m_sqrt = v0; //set the RMS speed as the virial speed as well
    v2m = v2m_sqrt*v2m_sqrt;
    v_epsilon = v0 / (lambda4 * N_particles);

    ln_Lambda = std::log(lambda4 * N_particles);
    n0 = (3.*N_particles) / (4.*M_PI * R0*R0*R0);
    Diffu_0_unit = (4.*M_PI * G*G * M_total*M_total * n0) / (3. * N_particles*N_particles * v0);
    Diffu_ref = diffusion_reference_value_Diffu_ref();
    coef_Diffu_parallel_iso = 2.*lambda1*lambda2*lambda2*lambda3 * std::sqrt(6./M_PI)*GXX * Diffu_0_unit / N_samples;
    coef_Diffu_tensor_uniform = 3. * ln_Lambda * Diffu_0_unit / N_samples;

    coef_const_Diffu = calculate_coef_const_Diffu();
    coef_const_Diffu_vel = calculate_coef_const_Diffu_vel();
    coef_Diffu_separable = 2.*lambda1*lambda2*lambda2*lambda3 * Diffu_0_unit;
    ratio_Diffu_old = Rm*Rm * coef_Diffu_parallel_iso / coef_const_Diffu;
    ratio_Diffu_old_vel = v2m_sqrt*coef_Diffu_tensor_uniform / coef_const_Diffu_vel;
}

void initialize_one_tensor(Diffu_tensor_vel& tensor){
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            tensor.D[i][j] = 0.;
}

void initialize_each_tensor(std::vector<Diffu_tensor_vel>& DT, size_t N){
    DT.resize(N);
    // for (size_t k = 0; k < N; ++k) {
    //     Diffu_tensor_vel& tensor = DT[k];
    //     initialize_one_tensor(tensor);
    // }
}

std::string getExecutablePath() {
    char buffer[255];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer)-1);
    if (len != -1) {
        buffer[len] = '\0';
        std::string fullPath(buffer);
        std::string exepath = fullPath.substr(0, fullPath.find_last_of('/'));
        std::cout<<"The current C++ exepath is: "<<exepath<<"\n";
        return exepath;
    } else {
        throw std::runtime_error("Cannot determine executable path");
    }
}

// Helper function: Compute the mean of a coordinate over all samples.
PrecisionSetting compute_mean(const std::vector<VelocitySpace>& velocities, int coordinate) {
    PrecisionSetting sum = 0.0;
    for (const auto& v : velocities) {
        if (coordinate == 0)
            sum += v.x;
        else if (coordinate == 1)
            sum += v.y;
        else if (coordinate == 2)
            sum += v.z;
        else
            throw std::runtime_error("Invalid coordinate index.");
    }
    return sum / velocities.size();
}

// Helper function: Compute the standard deviation of a coordinate over all samples.
PrecisionSetting compute_std(const std::vector<VelocitySpace>& velocities, int coordinate, PrecisionSetting mean) {
    PrecisionSetting sumsq = 0.0;
    for (const auto& v : velocities) {
        PrecisionSetting diff = 0.0;
        if (coordinate == 0)
            diff = v.x - mean;
        else if (coordinate == 1)
            diff = v.y - mean;
        else if (coordinate == 2)
            diff = v.z - mean;
        else
            throw std::runtime_error("Invalid coordinate index.");
        sumsq += diff * diff;
    }
    return std::sqrt(sumsq / velocities.size());
}

void readjust_velocities_dispersion_diag(
    std::vector<VelocitySpace>& velocities, const std::vector<PrecisionSetting>& vmean_3,
    const std::vector<PrecisionSetting>& dispersion_3
){
    // Check that the provided target vectors have size 3.
    if (vmean_3.size() != 3 || dispersion_3.size() != 3)
        throw std::runtime_error("vmean_3 and dispersion_3 must have exactly 3 elements each.");
    
    // Compute current mean and standard deviation (per coordinate)
    std::vector<PrecisionSetting> current_mean(3), current_std(3);
    for (int i = 0; i < 3; ++i) {
        current_mean[i] = compute_mean(velocities, i);
        current_std[i] = compute_std(velocities, i, current_mean[i]);
        
        // If the standard deviation is zero, avoid division by zero.
        if (current_std[i] == 0.0)
            current_std[i] = 1.0;
    }
    
    // Adjust each sample, coordinate-wise.
    for (auto& v : velocities) {
        // For x coordinate:
        v.x = vmean_3[0] + (v.x - current_mean[0]) * (dispersion_3[0] / current_std[0]);
        // For y coordinate:
        v.y = vmean_3[1] + (v.y - current_mean[1]) * (dispersion_3[1] / current_std[1]);
        // For z coordinate:
        v.z = vmean_3[2] + (v.z - current_mean[2]) * (dispersion_3[2] / current_std[2]);
    }
}
