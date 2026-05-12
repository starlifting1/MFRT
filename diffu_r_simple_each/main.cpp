#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include "diffu.h"
#include "generate_samples.h"
#include "direct_summation.h"
#include "Tree_method.h"
#include "tree_vel.h"
#include "fractal_dim.h"
// #include <sys/mman.h> //mmap



/*  The main function.
*/
int main(int argc, char* argv[]){
    //// Step 1. prepare and load
    // Read input arguments
    if (argc < 8) {
        std::cerr<<"Usage: <./main.exe> <samplesbinay> <N_particles> <M_total> <R0> <is_by_generating> <is_diffu> <N_iter>\n";
        std::cerr<<"Too less input argument. Exit.\n";
        return 1;
    }

    std::string samples_folder = getExecutablePath()+"/../data/samples_pos/";
    // std::string samples_folder = getExecutablePath()+"/../data/samples_vel/";
    // std::string samples_folder = getExecutablePath()+"/../data/samples_simulated/";
    std::string samplesbinay = argv[1]; //filename tag for read file or generated file
    std::string samples_read_file = samples_folder+"xv_"+samplesbinay+".txt";
    // std::string samples_read_file = samples_folder+samplesbinay+".binary";
    std::string input_tmp = argv[2];
    PrecisionSetting M_total_gal = PrecisionSetting(atof(argv[3]));
    PrecisionSetting R0_setting = PrecisionSetting(atof(argv[4]));
    bool is_by_generating = bool(atoi(argv[5]));
    bool is_diffu = bool(atoi(argv[6]));
    int N_iter = atoi(argv[7]);
    std::stringstream sstream(input_tmp);
    sstream>>N_particles; //constants
    // std::cout<<N_particles<<"\n";

    std::vector<PositionSpace> pos;
    std::vector<VelocitySpace> vel;

    if(!is_by_generating){ //sample points by reading instead of generating
        // if(!read_xv_from_binary(samples_read_file, pos, vel, N_particles)){
        //     std::cout<<"Can not open file. Exit.\n";
        //     return 1;
        // }
        if(!load_xv_from_txt(pos, vel, samples_read_file)){
            std::cout<<"Can not open file. Exit.\n";
            return 1;
        }
        N_particles = pos.size(); // update N_particles
    }
    N_samples = N_particles; //constants

    std::string suffix = "";
    suffix += "_read"+std::to_string(!is_by_generating);
    std::string tag_N = "_N"+std::to_string(N_particles);
    std::string tag_type, file_nr_type;

    std::string binary_file_uniform = samples_folder+"pos_uniform"+tag_N+".binary";
    std::string txt_file_uniform = samples_folder+"pos_uniform"+tag_N+".txt";
    std::string binary_file_noised = samples_folder+"pos_noised"+std::to_string(N_iter)+tag_N+".binary";
    std::string txt_file_noised = samples_folder+"pos_noised"+std::to_string(N_iter)+tag_N+".txt";

    // Calculate galaxy info
    M_total = M_total_gal*frac_mass; //constants
    R0 = R0_setting; //constants

    // b90 = calculate_b90(); //constants
    // v0 = calculate_v0(); //constants
    // coef_const_Diffu = calculate_coef_const_Diffu(); //constants

    // v2m_sqrt = v0; //constants //set the RMS speed as the virial speed
    // v2m = v2m_sqrt*v2m_sqrt; //constants
    // v_epsilon = calculate_v_epsilon(); //constants
    // coef_const_Diffu_vel = calculate_coef_const_Diffu_vel(); //constants
    
    calculate_fixed_const_coefs();
    print_galaxy_setting_info("init");

    double rate_calculate = 1.e4/N_particles; //to let count of targets be 1e4

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



    //// Step 2. generate position samples
    double start_time, end_time;

    if(!is_by_generating){ //particles by read gadget2 snapshot binary file
        
        N_particles = pos.size();
        print_particles_info(pos);
        print_all_particles(pos, vel, size_t(N_particles/10));
        readjust_positions(pos, Rm);
        Rm = get_mean_radius(pos);
        std::cout<<"mean_radius = "<<Rm<<"\n";
        
        tag_type = "read";
        std::string samples_read_file_normalized = samples_folder+"xv_normalized_"+samplesbinay+".txt";
        std::string file_nr_read = samples_folder+"nr_pos_"+samplesbinay+tag_N+".txt";
        write_xv_to_txt(pos, vel, samples_read_file_normalized);
        calculate_frac_dim(pos, Rm, file_nr_read, rate_calculate);

    }else{ //particles by generating
        tag_type = "generate";
        
        //// uniform
        start_time = omp_get_wtime();
        // generate_spherical_samples(N_particles, pos);
        generate_simple_xv(pos, vel, N_particles, Rm, vmean_3, dispersion_3_iso);
        readjust_positions(pos, Rm);
        Rm = get_mean_radius(pos);
        std::cout<<"mean_radius = "<<Rm<<"\n";

        write_xv_to_binary(pos, vel, binary_file_uniform);
        write_xv_to_txt(pos, vel, txt_file_uniform);
        end_time = omp_get_wtime();
        std::cout << "Time used of generating uniform is " << (end_time - start_time) << " seconds." << std::endl;
        
        // std::string file_nr_uniform = samples_folder+"nr_pos_uniform"+tag_N+".txt";
        // calculate_frac_dim(pos, Rm, file_nr_uniform, rate_calculate);

        //// noised
        start_time = omp_get_wtime();
        generate_noised_samples(pos, N_iter);
        suffix += "_noised"+std::to_string(N_iter);

        N_particles = pos.size();
        print_particles_info(pos);
        print_all_particles(pos, vel, size_t(N_particles/10));
        readjust_positions(pos, Rm);
        Rm = get_mean_radius(pos);
        std::cout<<"mean_radius = "<<Rm<<"\n";

        write_xv_to_binary(pos, vel, binary_file_noised);
        write_xv_to_txt(pos, vel, txt_file_noised);
        end_time = omp_get_wtime();
        std::cout << "Time used of noised motion is " << (end_time - start_time) << " seconds." << std::endl;
        
        std::string file_nr_noised = samples_folder+"nr_pos_"+samplesbinay+tag_N+".txt";
        calculate_frac_dim(pos, Rm, file_nr_noised, rate_calculate);
    }



    //// Step 3. calculate diffusion coef
    std::vector<PrecisionSetting> D_direct; //(N_particles, 0.0f);
    std::vector<PrecisionSetting> D_tree; //(N_particles, 0.0f);
    size_t N_calculate = N_particles;
    // const size_t N_calculate = size_t(N_particles/10);
    size_t N_display = size_t(N_particles/10);
    // size_t N_display = size_t(N_particles/100);
    Diffu_info Diffu_statistics;

    std::string file_save_binary = samples_folder+"Diffu_pos_"+samplesbinay+tag_N+".binary";
    std::string file_save_txt = samples_folder+"Diffu_pos_"+samplesbinay+tag_N+".txt";
    std::string file_statistics = samples_folder+"DiffuStatistics_pos_"+samplesbinay+tag_N+".txt";
    std::string file_pers = samples_folder+"DiffuPers_pos_"+samplesbinay+tag_N+".txt";

    // suffix += "_N"+std::to_string(N_particles);
    // std::string file_save_binary = samples_folder+samplesbinay+"_Diffur"+suffix+".binary";
    // std::string file_save_txt = samples_folder+samplesbinay+"_Diffur"+suffix+".txt";
    // std::string file_statistics = samples_folder+samplesbinay+"_Diffustatistics"+suffix+".txt";
    // std::string file_pers = samples_folder+samplesbinay+"_Diffu_pers"+suffix+".txt";
    
    if(is_diffu){
        D_direct.resize(N_particles, 0.0);
        D_tree.resize(N_particles, 0.0);

        //// (1) Run Tree Method
        std::cout << "\nRunning Tree Method..." << std::endl;
        Octree tree;
        tree.build(pos);

        start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < N_calculate; ++i) {
            D_tree[i] = tree.calculateDiffusionCoefficient(pos[i]);
            if(i%N_display==0){std::cout<<"Diffu: "<<i<<": "<<std::setprecision(10)<<D_tree[i]<<"\n";}
        }
        end_time = omp_get_wtime();
        std::cout << "Tree Method completed in " << (end_time - start_time) << " seconds." << std::endl;

        // bool is_enable_directsummation = true;
        bool is_enable_directsummation = false;
        if(is_enable_directsummation){
        //// (2) Run Direct Summation Method
        std::cout << "\nRunning Direct Summation Method..." << std::endl;
        start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < N_calculate; ++i) {
            D_direct[i] = directSummation_diffusionCoefficient_pos(pos, pos[i], i);
            if(i%N_display==0){std::cout<<"Diffu: "<<i<<": "<<std::setprecision(10)<<D_direct[i]<<"\n";}
        }
        end_time = omp_get_wtime();
        std::cout << "Direct Summation completed in " << (end_time - start_time) << " seconds." << std::endl;
        }

        //// (3) Save diffu to file
        //: after running DS method
        Diffu_statistics = calculate_diffu_statistics(D_direct, file_statistics);
        print_diffu_info(Diffu_statistics);

        //: after running tree method, the back one would be record
        Diffu_statistics = calculate_diffu_statistics(D_tree, file_statistics);
        print_diffu_info(Diffu_statistics);

        save_Diffur_to_binary(file_save_binary, D_direct, D_tree);
        save_Diffur_to_txt(file_save_txt, D_direct, D_tree); //big file

        //// (4) calculate diffur percentile
        // read_Diffur_from_binary(file_save_binary, D_direct, D_tree); //if calculated before
        auto diffu_pers = np_percentile(D_tree, qpers_display);
        save_percentile_to_txt(file_pers, qpers_display, diffu_pers);
    }

    return 0;
}



//     //Another version of main
//     //// Step 1. prepare and load
//     // Read input arguments
//     if (argc < 14) {
//         std::cerr<<"Usage: <./main.exe> <samplesbinay> <N_particles> <M_total> <R0> <is_by_generating> <is_diffu> <N_iter> <...>\n";
//         std::cerr<<"Too less input argument. Exit.\n";
//         return 1;
//     }

//     //// path
//     std::string samples_folder = getExecutablePath()+"/../data/samples_pos/";
//     std::string samplesbinay = argv[1];
//     std::string samples_read_file = samples_folder+samplesbinay+".binary";
//     std::string input_tmp = argv[2];
//     PrecisionSetting M_total_gal = PrecisionSetting(atof(argv[3]));
//     PrecisionSetting R0_setting = PrecisionSetting(atof(argv[4]));
//     bool is_by_generating = bool(atoi(argv[5]));
//     bool is_diffu = bool(atoi(argv[6]));
//     int N_iter = atoi(argv[7]);
//     std::stringstream sstream(input_tmp);
//     sstream>>N_particles; //constants
//     N_samples = N_particles; //constants

//     std::string suffix = "";
//     suffix += "_read"+std::to_string(!is_by_generating);
//     std::string tag_N = "_N"+std::to_string(N_particles);
//     std::string tag_type, file_nr_type;
    
//     std::string tag_for_pos = "pos_";
//     std::string tag_for_vel = "vel_";
//     std::string tag_uniform = "uniform";
//     std::string binary_file_uniform = samples_folder+tag_for_pos+tag_uniform+tag_N+".binary";
//     std::string txt_file_tag_uniform = samples_folder+tag_for_pos+tag_uniform+tag_N+".txt";
//     std::string file_nr_tag_uniform = samples_folder+"nr_"+tag_for_pos+tag_uniform+tag_N+".txt";
//     std::string tag_noised = "noised";
//     std::string binary_file_noised = samples_folder+tag_for_pos+tag_noised+tag_N+".binary";
//     std::string txt_file_noised = samples_folder+tag_for_pos+tag_noised+tag_N+".txt";
//     std::string file_nr_noised = samples_folder+"nr_"+tag_for_pos+tag_noised+tag_N+".txt";

//     //// basic setting
//     M_total = M_total_gal*frac_mass; //constants
//     R0 = R0_setting; //constants
//     b90 = calculate_b90(); //constants
//     v0 = calculate_v0(); //constants //set the typical cross speed as the virial speed
//     coef_const_Diffu = calculate_coef_const_Diffu(); //constants
    
//     v2m_sqrt = v0; //constants //set the RMS speed as the virial speed
//     v2m = v2m_sqrt*v2m_sqrt; //constants
//     v_epsilon = calculate_v_epsilon(); //constants
//     coef_const_Diffu_vel = calculate_coef_const_Diffu_vel(); //constants
//     print_galaxy_setting_info("init");

//     vmean_3 = { //the setted three mean velocity of velocity DF, usually zeros
//         PrecisionSetting(atof(argv[8])), PrecisionSetting(atof(argv[9])), PrecisionSetting(atof(argv[10]))
//     };
//     dispersion_3_iso_ratio = {1.0, 1.0, 1.0};
//     dispersion_3_high_ratio = { //the setted diag-component dispersion (about mean square speed) of velocity DF, usually refering v0
//         PrecisionSetting(atof(argv[11])), PrecisionSetting(atof(argv[12])), PrecisionSetting(atof(argv[13]))
//     };

//     std::vector<PrecisionSetting> dispersion_3_iso = rescale_dispersion_keep_ratio(vmean_3, dispersion_3_iso_ratio, v2m_sqrt); // Rescale dispersion vectors
//     std::vector<PrecisionSetting> dispersion_3_high = rescale_dispersion_keep_ratio(vmean_3, dispersion_3_high_ratio, v2m_sqrt);
//     print_vectors(dispersion_3_iso, dispersion_3_high);



// //// old exe usage
// // cd ~/workroom/0prog/proj2/MFRT/diffu_r_simple_each/
// // make; 
// // ./out.exe snapshot_fractal 10000 0 
// // ./out.exe snapshot_fractal 10000 14 
// int main1(int argc, char* argv[]){
//     std::string samples_folder = "../data/samples_simulated/";
//     std::string samplesbinay = argv[1];
//     bool is_by_generating = 1;
//     std::string input_tmp = argv[2];
//     std::stringstream sstream(input_tmp);
//     sstream>>N_particles; //constants
//     int N_iter = atoi(argv[3]);

//     std::string suffix = "";
//     suffix += "_read"+std::to_string(!is_by_generating);
//     suffix += "_noised"+std::to_string(N_iter);
//     suffix += "_N"+std::to_string(N_particles);
//     std::string file_save_binary = samples_folder+samplesbinay+"_Diffur"+suffix+".binary";
//     std::string file_save_txt = samples_folder+samplesbinay+"_Diffur"+suffix+".txt";
//     std::string file_statistics = samples_folder+samplesbinay+"_Diffustatistics"+suffix+".txt";
//     std::string file_pers = samples_folder+samplesbinay+"_Diffu_pers"+suffix+".txt";

//     std::vector<PrecisionSetting> D_direct;
//     std::vector<PrecisionSetting> D_tree;
//     // D_direct.resize(N_particles, 0.0f);
//     // D_tree.resize(N_particles, 0.0f);
//     if(!read_Diffur_from_binary(file_save_binary, D_direct, D_tree)){
//         exit(0);
//     }

//     std::vector<PrecisionSetting> qpers_display;
//     qpers_display = {
//         0.0,    0.0001, 0.001,  0.01,   0.1, 
//         0.2,    0.3,    0.4,    0.5,    0.6,    0.7,    0.8, 
//         0.9,    0.99,   0.999,  0.9999, 1.0
//     };
//     auto diffu_pers = np_percentile(D_tree, qpers_display);
//     save_percentile_to_txt(file_pers, qpers_display, diffu_pers);
//     return 0;
// }

// //// other code
//             //() to check for using deltaij
//             Diffu_tensor_vel Dd;
//             directSummationVelDiffusion_deltaij(vel, vel[i], i, Dd);
//             if(i%N_display==0){std::cout<<"Diffu: "<<i<<"\t: "<<std::setprecision(10)<<Dd.D[0][0]<<"\n";}
//             bool is_consist = check_value_consistant_rate(1, D_direct[i].D[0][0], Dd.D[0][0], i); //wrong
//             // bool is_consist = check_value_consistant_rate(1, D_direct[i].D[1][1], Dd.D[1][1], i); //wrong
//             // bool is_consist = check_value_consistant_rate(1, D_direct[i].D[1][0], Dd.D[1][0], i); //OK
//             if(!is_consist){
//                 print_scalar(i);
//                 print_Diffu_tensor_vel(D_direct[i]);
//                 print_Diffu_tensor_vel(Dd);
//                 exit(0);
//             }
