#include <iostream>
#include <omp.h>
#include "generate_gadget2_IC.h"

//// exe usage
// cd ~/workroom/0prog/proj2/MFRT/diffu_r_simple_each/generate_gadget2_IC/
// make; ./generate_IC.exe galaxy_general 10000 137. 0 0 XXX 
// make; ./generate_IC.exe galaxy_general -1 137. 50. 1 1 snapshot_011 
// make; ./generate_IC.exe galaxy_general -1 137. 50. 1 0 snapshot_011 

int main(int argc, char* argv[]) {

    //// Step 1. Prepare and load
    if (argc < 8) {
        std::cerr<<"Usage: <./generate_IC.exe> <samplesname> <N_generate> <M_total> <is_only_read_IC> <is_regenerate>\n";
        std::cerr<<"Too less input argument. Exit.\n";
        return 1;
    }

    std::string samplesname = argv[1];
    std::string binary_file_write = "../../data/samples_simulated/"+samplesname;
    std::string input_tmp = argv[2];
    size_t N_generate;
    std::stringstream sstream(input_tmp);
    sstream>>N_generate; //user setting
    PrecisionSetting M_total_IC = PrecisionSetting(atof(argv[3])); //user setting
    PrecisionSetting mean_radius = PrecisionSetting(atof(argv[4])); //user setting
    PrecisionSetting mass = M_total_IC/N_generate;
    int type_particle = 0; //user setting

    std::vector<PositionSpace> pos;
    std::vector<VelocitySpace> vel;
    double start_time, end_time;

    bool is_only_read_IC = bool(atoi(argv[5]));
    bool is_regenerate = bool(atoi(argv[6]));
    std::string snapshotname = argv[7];
    std::string binary_file_read = "../../data/samples_simulated/"+snapshotname;
    if(is_only_read_IC){ //by read snapshot
        
        load_gadget2_binary_to_xvCartesian(binary_file_read, pos, vel, N_generate);
        std::cout << "Convert gadget2 snapshot, done.\n";

        N_generate = pos.size();
        print_particles_info(pos);
        print_particles_info(vel);
        // print_all_particles(pos, vel, size_t(N_generate/10));

        if(is_regenerate){
            int n_clusters = 2;
            change_xv_to_multi_big_clusters(pos, vel, n_clusters);
            write_xvCartesian_to_binary_and_IC(binary_file_write, pos, vel, M_total/N_generate, type_particle);
            
            write_xv_to_binary(pos, vel, binary_file_write+".g1"+".binary");
            convert_gadget2_binary_to_txt(binary_file_write+".g1"); //if python plot
        }else{
            print_all_particles(pos, vel, size_t(N_generate/10));
            write_xv_to_binary(pos, vel, binary_file_read+".binary");
            convert_gadget2_binary_to_txt(binary_file_read); //if python plot
        }
    }else{ //by independent generating
        
        start_time = omp_get_wtime();
        std::cout << "Start to generate IC.\n";
        generate_simple_xv(pos, vel, N_generate, mean_radius, mean_3, dispersion_3);
        std::cout<< "pos[0].x = "<<pos[0].x<<", vel[0].x = "<<vel[0].x<<"\n";
        end_time = omp_get_wtime();
        std::cout << "Time used of generating IC is " << (end_time - start_time) << " seconds." << std::endl;

        write_xvCartesian_to_binary_and_IC(binary_file_write, pos, vel, mass, type_particle);
        std::cout << "Write binary and txt files, done.\n";

        print_particles_info(pos);
        print_particles_info(vel);
        // print_all_particles(pos, vel, size_t(N_generate/10));
    }
    return 0;
}
