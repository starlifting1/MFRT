#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include "diffu.h"
#include "generate_samples.h"
#include "direct_summation.h"
#include "Tree_method.h"
#include "tree_vel.h"
#include "whitening_recoloring.h"
#include "fractal_dim.h"
// #include <sys/mman.h> //mmap



/*  To calculate diffusion coef for each sample.
*/
void calculate_diffu_vel_workflow(
    std::vector<Diffu_tensor_vel>& D_direct, std::vector<Diffu_tensor_vel>& D_tree, 
    const std::vector<VelocitySpace>& vel, 
    std::string samples_folder, std::string tag_type, std::string tag_N, 
    size_t N_calculate, bool is_enable_directsummation=false, size_t N_display=10
){
    //// (1) Run Tree Method
    std::cout << "\nRunning Tree Method..." << std::endl;
    Octree_vel tree;
    tree.build(vel);
    size_t N_each_display = N_calculate/N_display;

    double start_time, end_time;
    start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < N_calculate; ++i) {
        tree.calculateDiffusionCoefficient(vel[i], D_tree[i]);
        if(i%N_each_display==0){
            std::cout<<"Target "<<i<<"\t, "; //<<std::setprecision(10)<<D_tree[i].D[0][0]<<"\n";
            print_Diffu_tensor_vel(D_tree[i]);
        }
    }
    end_time = omp_get_wtime();
    std::cout << "Tree Method completed in " << (end_time - start_time) << " seconds." << std::endl;

    //// (2) Run Direct Summation Method
    if(is_enable_directsummation){
    std::cout << "\nRunning Direct Summation Method..." << std::endl;
    start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
    for(size_t i = 0; i < N_calculate; ++i) {
        directSummationVelDiffusion_fraction(vel, vel[i], i, D_direct[i]);
        if(i%N_each_display==0){
            std::cout<<"Diffu: "<<i<<"\t: "<<std::setprecision(10)<<D_direct[i].D[0][0]<<"\n";
            print_Diffu_tensor_vel(D_direct[i]);
        }
    }
    end_time = omp_get_wtime();
    std::cout << "Direct Summation completed in " << (end_time - start_time) << " seconds." << std::endl;
    }

    //// (3) Save diffu to file
    Diffu_info Diffu_statistics;
    std::string file_save_binary = samples_folder+"Diffu_vel_"+tag_type+tag_N+".binary";
    std::string file_save_txt = samples_folder+"Diffu_vel_"+tag_type+tag_N+".txt"; //??
    std::string file_statistics = samples_folder+"DiffuStatistics_vel_"+tag_type+tag_N+".txt"; //record N_iter
    std::string file_pers = samples_folder+"DiffuPers_vel_"+tag_type+tag_N+".txt";

    // //: after running directsummation method
    // std::vector<PrecisionSetting> diffu_eff1;
    // Statistics_info Sta1 = calculate_diffu_statistics(D_direct, diffu_eff1);
    // save_diffu_statistics(file_statistics, Sta1);

    //: after running tree method
    std::vector<PrecisionSetting> diffu_eff;
    Statistics_info Sta = calculate_diffu_statistics(D_tree, diffu_eff);
    save_diffu_statistics(file_statistics, Sta);

    //: save all calculated diffu data
    save_all_diffu_tensors_to_binary(file_save_binary, D_tree);
    save_all_diffu_tensors_to_txt(file_save_txt, D_tree);

    //: calculate diffur percentile
    auto diffu_pers = np_percentile(diffu_eff, qpers_display);
    save_percentile_to_txt(file_pers, qpers_display, diffu_pers);
}



/*  To generate samples from a Gaussian-core and powerlaw-tail DF.
*/
void generate_tail_DF(
    std::vector<VelocitySpace>& vel, size_t N_particles, PrecisionSetting v2m_sqrt, std::vector<PrecisionSetting> vmean_3, 
    std::vector<PrecisionSetting> dispersion_3_iso, std::vector<PrecisionSetting> dispersion_3_high, 
    bool is_use_iso=true, bool is_print=false
){
    // //: These parameters are fitted from simulation data, adjusted.
    // PrecisionSetting sigma = 143.02621867679355;  // Gaussian dispersion for the core
    // PrecisionSetting vc = 293.1071270852349;        // Tail scale for power-law part
    // PrecisionSetting gamma = 4.69149629070613;        // Tail exponent
    // // PrecisionSetting k = 0.08853016313884869;         // Logistic steepness
    // PrecisionSetting k = 0.06;         // Logistic steepness
    // PrecisionSetting v1 = 396.99512203793194;         // Logistic transition speed
    // PrecisionSetting A_pl = 0.311190198455538;         // Tail amplitude
    // PrecisionSetting epsilon = 0.1;                   // Tail translation parameter
    
    //: These parameters are fitted from simulation data, old.
    PrecisionSetting sigma = 143.02621867679355;  // Gaussian dispersion for the core
    PrecisionSetting vc = 293.1071270852349;        // Tail scale for power-law part
    PrecisionSetting gamma = 4.69149629070613;        // Tail exponent
    PrecisionSetting k = 0.08853016313884869;         // Logistic steepness
    // PrecisionSetting k = 0.06;         // Logistic steepness
    PrecisionSetting v1 = 396.99512203793194;         // Logistic transition speed
    PrecisionSetting A_pl = 0.311190198455538;         // Tail amplitude
    PrecisionSetting epsilon = 0.1;                   // Tail translation parameter
    
    //: scale and dispersion setting
    std::vector<PrecisionSetting> dispersion_3_core;
    std::vector<PrecisionSetting> dispersion_3_tail;
    if(is_use_iso){
        dispersion_3_core = rescale_dispersion_keep_ratio(vmean_3, dispersion_3_iso, sigma*std::sqrt(3.));
        dispersion_3_tail = rescale_dispersion_keep_ratio(vmean_3, dispersion_3_iso, vc*std::sqrt(3.));
    }else{
        dispersion_3_core = rescale_dispersion_keep_ratio(vmean_3, dispersion_3_high, sigma*std::sqrt(3.));
        dispersion_3_tail = rescale_dispersion_keep_ratio(vmean_3, dispersion_3_high, vc*std::sqrt(3.));
    }

    PrecisionSetting vmax = v2m_sqrt * 5.0;
    std::vector<PrecisionSetting> vmax_three(dispersion_3_core.size());
    for (size_t i = 0; i < dispersion_3_core.size(); ++i) {
        vmax_three[i] = dispersion_3_core[i] * 5.0;
    }
    
    //: Isotropic DF sample using combined DF:
    auto vel_GP_iso = sample_velocity_isotropic_df(
        N_particles, sigma, vc, gamma, 
        k, v1, A_pl, epsilon, vmax
    );
    
    //: Convert the isotropic sample to an anisotropic sample (1D->3D conversion):
    auto vel_GP_aniso_13 = sample_isotropic_to_anisotropic(
        vel_GP_iso, dispersion_3_core
    );
    
    //: Sample anisotropic DF from the formula:
    auto vel_GP_aniso_3D = sample_anisotropic_df(
        N_particles, dispersion_3_core, dispersion_3_tail, gamma, 
        k, v1, A_pl, epsilon, vmax_three
    );
    
    //: Print the first 5 samples from each method to check (for debugging).
    if(is_print){
        std::cout << "First 5 samples from vel_GP_iso:" << std::endl;
        for (size_t i = 0; i < std::min(N_particles, size_t(5)); ++i) {
            std::cout << vel_GP_iso[i].x << " " << vel_GP_iso[i].y << " " << vel_GP_iso[i].z << std::endl;
        }
        std::cout << "\nFirst 5 samples from vel_GP_aniso_13:" << std::endl;
        for (size_t i = 0; i < std::min(N_particles, size_t(5)); ++i) {
            std::cout << vel_GP_aniso_13[i].x << " " << vel_GP_aniso_13[i].y << " " << vel_GP_aniso_13[i].z << std::endl;
        }
        std::cout << "\nFirst 5 samples from vel_GP_aniso_3D:" << std::endl;
        for (size_t i = 0; i < std::min(N_particles, size_t(5)); ++i) {
            std::cout << vel_GP_aniso_3D[i].x << " " << vel_GP_aniso_3D[i].y << " " << vel_GP_aniso_3D[i].z << std::endl;
        }
    }
    
    //: copy to vel
    copy_xv_sample(vel_GP_aniso_3D, vel);
}



/*  The main function.
*/
int main(int argc, char* argv[]){
    //// Step 1. prepare and load
    // Read input arguments
    if (argc < 14) {
        std::cerr<<"Usage: <./main.exe> <samplesbinay> <N_particles> <M_total> <R0> <is_by_generating> <is_diffu> <N_iter> <...>\n";
        std::cerr<<"Too less input argument. Exit.\n";
        return 1;
    }

    //// path
    std::string samples_folder = getExecutablePath()+"/../data/samples_vel/";
    std::string samplesbinay = argv[1];
    std::string samplesbinay_file = samples_folder+samplesbinay+".binary";
    std::string input_tmp = argv[2];
    PrecisionSetting M_total_gal = PrecisionSetting(atof(argv[3]));
    PrecisionSetting R0_setting = PrecisionSetting(atof(argv[4]));
    bool is_by_generating = bool(atoi(argv[5]));
    bool is_diffu = bool(atoi(argv[6]));
    int N_iter = atoi(argv[7]);
    std::stringstream sstream(input_tmp);
    sstream>>N_particles; //constants
    N_samples = N_particles; //constants

    std::string suffix = "";
    suffix += "_read"+std::to_string(!is_by_generating);
    std::string tag_N = "_N"+std::to_string(N_particles);
    std::string tag_type, file_nr_type;
    
    std::string tag_for_pos = "pos_";
    std::string tag_for_vel = "vel_";
    std::string tag_iso = "iso";
    std::string binary_file_iso = samples_folder+"vel_"+tag_iso+tag_N+".binary";
    std::string txt_file_iso = samples_folder+"vel_"+tag_iso+tag_N+".txt";
    std::string file_nr_iso = samples_folder+"nr_vel_"+tag_iso+tag_N+".txt";
    std::string tag_aniso = "aniso";
    std::string binary_file_aniso = samples_folder+"vel_"+tag_aniso+tag_N+".binary";
    std::string txt_file_aniso = samples_folder+"vel_"+tag_aniso+tag_N+".txt";
    std::string file_nr_aniso = samples_folder+"nr_vel_"+tag_aniso+tag_N+".txt";
    std::string tag_tail = "tail";
    std::string binary_file_tail = samples_folder+"vel_"+tag_tail+tag_N+".binary";
    std::string txt_file_tail = samples_folder+"vel_"+tag_tail+tag_N+".txt";
    std::string file_nr_tail = samples_folder+"nr_vel_"+tag_tail+tag_N+".txt";
    std::string tag_noised = "noised";
    std::string binary_file_noised = samples_folder+"vel_"+tag_noised+tag_N+".binary";
    std::string txt_file_noised = samples_folder+"vel_"+tag_noised+tag_N+".txt";
    std::string file_nr_noised = samples_folder+"nr_vel_"+tag_noised+tag_N+".txt";
    std::string tag_composite = "composite";
    std::string binary_file_composite = samples_folder+"vel_"+tag_composite+tag_N+".binary";
    std::string txt_file_composite = samples_folder+"vel_"+tag_composite+tag_N+".txt";
    std::string file_nr_composite = samples_folder+"nr_vel_"+tag_composite+tag_N+".txt";
    
    //// basic setting
    M_total = M_total_gal*frac_mass; //constants
    R0 = R0_setting; //constants

    // b90 = calculate_b90(); //constants
    // v0 = calculate_v0(); //constants //set the typical cross speed as the virial speed
    // coef_const_Diffu = calculate_coef_const_Diffu(); //constants
    
    // v2m_sqrt = v0; //constants //set the RMS speed as the virial speed
    // v2m = v2m_sqrt*v2m_sqrt; //constants
    // v_epsilon = calculate_v_epsilon(); //constants
    // coef_const_Diffu_vel = calculate_coef_const_Diffu_vel(); //constants
    
    calculate_fixed_const_coefs();
    print_galaxy_setting_info("init");

    double rate_calculate = 1.e4/N_particles; //to let count of targets be 1e4
    // bool is_enable_directsummation = true;
    bool is_enable_directsummation = false;

    vmean_3 = { //the setted three mean velocity of velocity DF, usually zeros
        PrecisionSetting(atof(argv[8])), PrecisionSetting(atof(argv[9])), PrecisionSetting(atof(argv[10]))
    };
    dispersion_3_iso_ratio = {1.0, 1.0, 1.0};
    dispersion_3_high_ratio = { //the setted diag-component dispersion (about mean square speed) of velocity DF, usually refering v0
        PrecisionSetting(atof(argv[11])), PrecisionSetting(atof(argv[12])), PrecisionSetting(atof(argv[13]))
    };

    std::vector<PrecisionSetting> dispersion_3_iso = rescale_dispersion_keep_ratio(vmean_3, dispersion_3_iso_ratio, v2m_sqrt); // Rescale dispersion vectors
    std::vector<PrecisionSetting> dispersion_3_high = rescale_dispersion_keep_ratio(vmean_3, dispersion_3_high_ratio, v2m_sqrt);
    print_vectors(dispersion_3_iso, dispersion_3_high);

    // Target mean velocity (for example, zero)
    Eigen::Vector3d targetMean(vmean_3[0], vmean_3[1], vmean_3[2]);
    
    // Target covariance matrix for the velocity distribution. We assume the off-diagonal elements should be 0.
    Eigen::Matrix3d cov_iso = Eigen::Matrix3d::Zero();
    cov_iso(0, 0) = std::pow(dispersion_3_iso[0],2);
    cov_iso(1, 1) = std::pow(dispersion_3_iso[1],2);
    cov_iso(2, 2) = std::pow(dispersion_3_iso[2],2);
    
    Eigen::Matrix3d cov_high = Eigen::Matrix3d::Zero();
    cov_high(0, 0) = std::pow(dispersion_3_high[0],2);
    cov_high(1, 1) = std::pow(dispersion_3_high[1],2);
    cov_high(2, 2) = std::pow(dispersion_3_high[2],2);

    //// Step 2. generate position samples and calculate diffusion coef
    std::vector<PositionSpace> pos;
    std::vector<VelocitySpace> vel_usual;
    std::vector<VelocitySpace> vel;

    std::vector<Diffu_tensor_vel> D_direct;
    std::vector<Diffu_tensor_vel> D_tree;
    size_t N_calculate = N_particles;
    // const size_t N_calculate = size_t(N_particles/10);
    double start_time, end_time;

    //// begin to run
    if(!is_by_generating){ //A. particles by read gadget2 snapshot binary file
        
        std::cout<<"\n\n\nRead positions from file and do not generate.\n";
        if(!read_xv_from_binary(samplesbinay_file, pos, vel, N_particles)){
            std::cout<<"Can not open file. Exit.\n";
            return 1;
        }

        N_particles = pos.size();
        print_particles_info(pos);
        print_all_particles(pos, vel, size_t(N_particles/10));
        readjust_positions(pos, Rm);
        Rm = get_mean_radius(pos);
        std::cout<<"mean_radius = "<<Rm<<"\n";
        
    }else{ //B. particles by generating

        //// a. isotropic Gaussian DF
        start_time = omp_get_wtime();
        tag_type = tag_iso;
        file_nr_type = file_nr_iso;
        generate_simple_xv(pos, vel_usual, N_particles, Rm, vmean_3, dispersion_3_iso); //pos uniform and vel Gaussian
        readjust_positions(pos, Rm);
        copy_xv_sample(vel_usual, vel);

        print_VelocitySampleInfo(vel, tag_type);
        // print_galaxy_setting_info(tag_type);
        write_xv_to_binary(pos, vel, binary_file_iso);
        write_xv_to_txt(pos, vel, txt_file_iso);
        end_time = omp_get_wtime();
        std::cout << "Time used of generating "<<tag_type<<" is " << (end_time - start_time) << " seconds." << std::endl;

        if(is_diffu){ //the Diffu_tensor_vel would be initialized to zero before calculate
            initialize_each_tensor(D_direct, N_calculate);
            initialize_each_tensor(D_tree, N_calculate);
        }
        if(is_diffu){
            calculate_frac_dim(vel, v2m_sqrt, file_nr_type, rate_calculate);
            calculate_diffu_vel_workflow(D_direct, D_tree, vel, samples_folder, tag_type, tag_N, N_calculate, is_enable_directsummation);
        }

        //// b. anisotropic Gaussian DF
        start_time = omp_get_wtime();
        tag_type = tag_aniso;
        file_nr_type = file_nr_aniso;
        copy_xv_sample(vel_usual, vel);
        // readjust_velocities_v2m_sqrt(vel, vmean_3, dispersion_3_high);
        // readjust_velocities_dispersion_diag(vel, vmean_3, dispersion_3_high);
        readjust_velocities_dispersion_affine(vel, targetMean, cov_high);
        
        print_VelocitySampleInfo(vel, tag_type);
        // print_galaxy_setting_info(tag_type);
        write_xv_to_binary(pos, vel, binary_file_aniso);
        write_xv_to_txt(pos, vel, txt_file_aniso);
        end_time = omp_get_wtime();
        std::cout << "Time used of generating "<<tag_type<<" is " << (end_time - start_time) << " seconds." << std::endl;

        if(is_diffu){
            calculate_frac_dim(vel, v2m_sqrt, file_nr_type, rate_calculate);
            calculate_diffu_vel_workflow(D_direct, D_tree, vel, samples_folder, tag_type, tag_N, N_calculate, is_enable_directsummation);
        }

        //// c. long-tailed DF
        start_time = omp_get_wtime();
        tag_type = tag_tail;
        file_nr_type = file_nr_tail;
        copy_xv_sample(vel_usual, vel);
        bool is_use_iso = true;
        generate_tail_DF(vel, N_particles, v2m_sqrt, vmean_3, dispersion_3_iso, dispersion_3_high, is_use_iso);
        if(is_use_iso){
            // readjust_velocities_dispersion_diag(vel, vmean_3, dispersion_3_iso);
            readjust_velocities_dispersion_affine(vel, targetMean, cov_iso);
        }else{
            // readjust_velocities_dispersion_diag(vel, vmean_3, dispersion_3_high);
            readjust_velocities_dispersion_affine(vel, targetMean, cov_high);
        }
        print_VelocitySampleInfo(vel, tag_type);
        // print_galaxy_setting_info(tag_type);
        write_xv_to_binary(pos, vel, binary_file_tail);
        write_xv_to_txt(pos, vel, txt_file_tail);
        end_time = omp_get_wtime();
        std::cout << "Time used of generating "<<tag_type<<" is " << (end_time - start_time) << " seconds." << std::endl;

        if(is_diffu){
            calculate_frac_dim(vel, v2m_sqrt, file_nr_type, rate_calculate);
            calculate_diffu_vel_workflow(D_direct, D_tree, vel, samples_folder, tag_type, tag_N, N_calculate, is_enable_directsummation);
        }

        //// d. fractal DF
        start_time = omp_get_wtime();
        tag_type = tag_noised;
        file_nr_type = file_nr_noised;
        copy_xv_sample(vel_usual, vel);
        generate_noised_samples(vel, N_iter);
        // readjust_velocities_dispersion_diag(vel, vmean_3, dispersion_3_iso);
        readjust_velocities_dispersion_affine(vel, targetMean, cov_iso);
        
        print_VelocitySampleInfo(vel, tag_type);
        // print_galaxy_setting_info(tag_type);
        write_xv_to_binary(pos, vel, binary_file_noised);
        write_xv_to_txt(pos, vel, txt_file_noised);
        end_time = omp_get_wtime();
        std::cout << "Time used of generating "<<tag_type<<" is " << (end_time - start_time) << " seconds." << std::endl;

        if(is_diffu){
            calculate_frac_dim(vel, v2m_sqrt, file_nr_type, rate_calculate); //??
            calculate_diffu_vel_workflow(D_direct, D_tree, vel, samples_folder, tag_type, tag_N, N_calculate, is_enable_directsummation);
        }

        //// e. composite modifications DF
        start_time = omp_get_wtime();
        tag_type = tag_composite;
        file_nr_type = file_nr_composite;
        copy_xv_sample(vel_usual, vel);
        is_use_iso = false;
        generate_tail_DF(vel, N_particles, v2m_sqrt, vmean_3, dispersion_3_iso, dispersion_3_high, is_use_iso);
        // readjust_velocities_dispersion_diag(vel, vmean_3, dispersion_3_iso);
        readjust_velocities_dispersion_affine(vel, targetMean, cov_iso);
        // // readjust_velocities_dispersion_diag(vel, vmean_3, dispersion_3_high);
        // readjust_velocities_dispersion_affine(vel, targetMean, cov_high);
        generate_noised_samples(vel, N_iter); //same x, y, z step
        // readjust_velocities_dispersion_diag(vel, vmean_3, dispersion_3_high);
        readjust_velocities_dispersion_affine(vel, targetMean, cov_high);

        print_VelocitySampleInfo(vel, tag_type);
        // print_galaxy_setting_info(tag_type);
        write_xv_to_binary(pos, vel, binary_file_composite);
        write_xv_to_txt(pos, vel, txt_file_composite);
        end_time = omp_get_wtime();
        std::cout << "Time used of generating "<<tag_type<<" is " << (end_time - start_time) << " seconds." << std::endl;
    
        if(is_diffu){
            calculate_frac_dim(vel, v2m_sqrt, file_nr_type, rate_calculate);
            calculate_diffu_vel_workflow(D_direct, D_tree, vel, samples_folder, tag_type, tag_N, N_calculate, is_enable_directsummation);
        }
    }

    return 0;
}
