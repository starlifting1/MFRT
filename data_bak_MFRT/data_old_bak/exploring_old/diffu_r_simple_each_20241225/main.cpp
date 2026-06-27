#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include "diffu.h"
#include "generate_samples.h"
#include "Tree_method.h"

// #include <fcntl.h>
// #include <sys/mman.h> //mmap
// #include <unistd.h>
// #include <sys/stat.h>



//// exe usage
// cd ~/workroom/0prog/proj2/MFRT/diffu_r_simple_each/
// make; ./out.exe uniform_nonoise 10000 137. 50. 1 1 0 
// make; ./out.exe uniform_noise 10000 137. 50. 1 1 14 
// make; ./out.exe uniform_noise 10000 137. 50. 1 1 30 
// make; ./out.exe snapshot_000.txt 1000000 137. 50. 0 1 0 

int main(int argc, char* argv[]) {
    //// Step 1. prepare and load
    // Read input arguments
    if (argc < 8) {
        std::cerr<<"Usage: <./main.exe> <samplesbinay> <N_particles> <M_total> <R0> <is_samples> <is_diffu> <N_iter>\n";
        std::cerr<<"Too less input argument. Exit.\n";
        return 1;
    }

    std::string samples_folder = "../data/samples_simulated/";
    std::string samplesbinay = argv[1];
    std::string samplesbinay_file = samples_folder+samplesbinay+".binary";
    std::string input_tmp = argv[2];
    float M_total_gal = float(atof(argv[3]));
    float R0_setting = float(atof(argv[4]));
    bool is_samples = bool(atoi(argv[5]));
    bool is_diffu = bool(atoi(argv[6]));
    int N_iter = atoi(argv[7]);

    std::string suffix = "";
    suffix += "_read"+std::to_string(!is_samples);
    std::string binary_file_uniform = samples_folder+"uniform.binary";
    // std::string txt_file_uniform = samples_folder+"uniform.txt";
    std::string binary_file_noised = samples_folder+"noised.binary";
    // std::string txt_file_noised = samples_folder+"noised.txt";
    double start_time, end_time;

    // Calculate galaxy info
    std::stringstream sstream(input_tmp);
    sstream>>N_particles; //constants
    // std::cout<<N_particles<<"\n";
    N_samples = N_particles; //constants
    M_total = M_total_gal*frac_mass; //constants
    R0 = R0_setting; //constants
    b90 = calculate_b90(); //constants
    v0 = calculate_v0(); //constants
    coef_const_Diffu = calculate_coef_const_Diffu(); //constants
    print_galaxy_setting_info();



    //// Step 2. generate position samples
    std::vector<PositionSpace> pos;
    std::vector<VelocitySpace> vel;
    // std::vector<PositionSpace> pos_no_use; //null
    // std::vector<xvSpaceCartesian> xvCartesian;

    if(!is_samples){ //particles by read gadget2 snapshot binary file
        
        std::cout<<"\n\n\nRead positions from file and do not generate.\n";
        if(!read_xv_from_binary(samplesbinay_file, pos, vel, N_particles)){
            std::cout<<"Can not open file. Exit.\n";
            return 1;
        }

        N_particles = pos.size();
        print_particles_info(pos);
        print_all_particles(pos, vel, size_t(N_particles/10));
        readjust_positions(pos, R0);
        R0 = get_mean_radius(pos);
        // R0 = get_mean_radius(pos, true);
        std::cout<<"mean_radius = "<<R0<<"\n";
    }else{ //particles by generating
        
        //// uniform
        start_time = omp_get_wtime();
        // generate_spherical_samples(N_particles, pos);
        generate_simple_xv(pos, vel, N_particles, R0, mean_3, dispersion_3);
        readjust_positions(pos, R0);
        R0 = get_mean_radius(pos);
        // R0 = get_mean_radius(pos, true);
        std::cout<<"mean_radius = "<<R0<<"\n";
        // write_samples_to_binary(pos, binary_file_uniform);
        // read_binary_and_convert_to_txt(binary_file_uniform, txt_file_uniform, pos_no_use);
        write_xv_to_binary(pos, vel, binary_file_uniform);
        end_time = omp_get_wtime();
        std::cout << "Time used of generating uniform is " << (end_time - start_time) << " seconds." << std::endl;

        //// noised
        start_time = omp_get_wtime();
        generate_noised_samples(pos, N_iter);
        suffix += "_noised"+std::to_string(N_iter);

        N_particles = pos.size();
        print_particles_info(pos);
        print_all_particles(pos, vel, size_t(N_particles/10));
        readjust_positions(pos, R0);
        R0 = get_mean_radius(pos);
        // R0 = get_mean_radius(pos, true);
        std::cout<<"mean_radius = "<<R0<<"\n";

        write_xv_to_binary(pos, vel, binary_file_noised);
        end_time = omp_get_wtime();
        std::cout << "Time used of noised motion is " << (end_time - start_time) << " seconds." << std::endl;
    }



    //// Step 3. calculate diffusion coef
    std::vector<float> D_direct(N_particles, 0.0f);
    std::vector<float> D_tree(N_particles, 0.0f);
    size_t N_calculate = N_particles;
    // const size_t N_calculate = size_t(N_particles/10);
    size_t N_display = size_t(N_particles/10);
    // size_t N_display = size_t(N_particles/100);

    if(is_diffu){
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

//         //// (2) Run Direct Summation Method
//         std::cout << "\nRunning Direct Summation Method..." << std::endl;
//         start_time = omp_get_wtime();
//         // directSummation(pos, D_direct, N_particles);
// #pragma omp parallel for schedule(dynamic)
//         for(size_t i = 0; i < N_calculate; ++i) {
//             D_direct[i] = calculateDiffusionCoefficient_DS(pos, pos[i], i);
//             if(i%N_display==0){std::cout<<"Diffu: "<<i<<": "<<std::setprecision(10)<<D_direct[i]<<"\n";}
//         }
//         end_time = omp_get_wtime();
//         std::cout << "Direct Summation completed in " << (end_time - start_time) << " seconds." << std::endl;

        //// (3) Save diffu to file
        Diffu_info Diffu_statistics;
        suffix += "_N"+std::to_string(N_particles);
        std::string file_save_binary = samples_folder+samplesbinay+"_Diffur"+suffix+".binary";
        std::string file_save_txt = samples_folder+samplesbinay+"_Diffur"+suffix+".txt";
        std::string file_statistics = samples_folder+samplesbinay+"_Diffustatistics"+suffix+".txt";
        Diffu_statistics = calculate_diffu_statistics(D_direct, file_statistics, false);
        print_diffu_info(Diffu_statistics);
        Diffu_statistics = calculate_diffu_statistics(D_tree, file_statistics);
        print_diffu_info(Diffu_statistics);

        save_Diffur_to_binary(file_save_binary, D_direct, D_tree);
        save_Diffur_to_txt(file_save_txt, D_direct, D_tree); //if python plot
    }

    return 0;
}
