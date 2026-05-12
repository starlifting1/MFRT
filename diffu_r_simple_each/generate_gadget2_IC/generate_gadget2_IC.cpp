/* 
    ================================================================
    To read from .txt (simple particles data or whole particles data) 
    and write to .g1; or as versa.
    ===================================================================
*/

#include "generate_gadget2_IC.h"

/*
# To read and write getget2 snapshot.
local_is_trans_txt=${1}
filename_brg=galaxy_general
if [ ${local_is_trans_txt} -eq 0 ]; then
    echo -e "from g1 to txt"
    ./read_snapshot.exe 2 ${filename_brg}.g1 #to get Input.txt and Input.txt.SCF from Input
    mv ${filename_brg}.g1.txt ${filename_brg}.txt
else
    echo -e "from txt to g1"
    ./read_snapshot.exe 1 galaxy_general.txt #to get Input.g1 and Input.g1.SCF from Input
    mv galaxy_general.txt.g1 galaxy_general.g1
fi
*/
int io_snapshot(int argc, char* argv[]){

    int tag_read_write = atoi(argv[1]); //1: read; other: write ...
    string path_snapshot = (argv[2]); //the name of target file
    RW_Snapshot RWSS;
    if(tag_read_write==1){ //from .txt
        // string path_snapshot1 = "galaxy_general1.g1";
        // RWSS.allocate_memory();
        // RWSS.load(path_snapshot1);
        // RWSS.NumPart = 100000; //??
        // for(int n=0; n<RWSS.NumPart; n++){
        //     RWSS.Id[n] = n;
        //     std::cout<<RWSS.Id[n]<<" ";
        // }
        // RWSS.print_header_info();
        // int id = 1;
        // cout<<RWSS.NumPartTot<<" "<<RWSS.NumPart<<" "
        // <<RWSS.header1.npart[1]<<" "<<RWSS.header1.npartTotal[1]<<" Here_11\n";

        std::cout<<RWSS.num<<" "<<RWSS.NumPart<<" "<<RWSS.NumPartTot<<", tot1\n";
        RWSS.is_reorder = 0;
        RWSS.allocate_memory(); //?? reordering
        RWSS.set_header();
        std::cout<<RWSS.num<<" "<<RWSS.NumPart<<" "<<RWSS.NumPartTot<<", tot2\n";
        RWSS.print_header_info("123");
        // RWSS.read_PD_txt(path_snapshot, 0); //cold IC //one does not use it unless adding a argv[3]
        RWSS.read_PD_txt(path_snapshot, 1); //modified IC
        std::cout<<RWSS.num<<" "<<RWSS.NumPart<<" "<<RWSS.NumPartTot<<", tot3\n";
        RWSS.set_header();
        RWSS.print_header_info("456");
        RWSS.write_gadget_ics_known(path_snapshot);
        // RWSS.print_header_info("789");
        // RWSS.write_PD_toSCF(path_snapshot, ".SCF");

        // // RWSS.allocate_memory();
        // path_snapshot += ".g1";
        // RWSS.load(path_snapshot);
        // RWSS.write_PD_txt(path_snapshot, ".txt");
    }else{ //from .g1
        RWSS.is_reorder = 0;
        RWSS.allocate_memory();
        RWSS.load(path_snapshot); //??
        RWSS.write_PD_txt(path_snapshot, ".txt");
        RWSS.write_PD_toSCF(path_snapshot, ".SCF");

        // // RWSS.allocate_memory();
        // path_snapshot += ".txt";
        // RWSS.read_PD_txt(path_snapshot, 1);
        // RWSS.set_header();
        // RWSS.write_gadget_ics_known(path_snapshot);
    }

    return 0;
}

/*  There might cause bugs because of size_t -> int.
*/
int write_xvCartesian_to_binary_and_IC(
    string path_snapshot, const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, 
    PrecisionSetting mass, int type, bool is_ID_from_1
){
    size_t N_generate = pos.size();
    if(N_generate!=vel.size()){
        std::cerr<<"Wrong size of pos data and vel data.\n";
        return 1;
    }

    RW_Snapshot RWSS;
    RWSS.NumPart = N_generate;
    RWSS.NumPartTot = N_generate;
    RWSS.is_reorder = 0;
    RWSS.allocate_memory(); //?? reordering

    RWSS.set_header();
    RWSS.print_header_info("123");
    RWSS.write_samples_to_PD(pos, vel);
    RWSS.set_header();
    RWSS.print_header_info("456");
    RWSS.write_gadget_ics_known(path_snapshot);

    // RWSS.write_samples_to_binary(pos, vel, path_snapshot+".binary");
    // RWSS.write_PD_txt(path_snapshot, ".txt");
    return 0;
}

int load_gadget2_binary_to_xvCartesian(string path_binary, std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N){
    RW_Snapshot RWSS;
    RWSS.is_reorder = 0;
    RWSS.allocate_memory();
    RWSS.load(path_binary); //??
    N = RWSS.NumPartTot;
    RWSS.write_PD_to_xvCartesian(pos, vel);
    return 0;
}

int convert_gadget2_binary_to_txt(string path_binary){
    RW_Snapshot RWSS;
    RWSS.is_reorder = 0;
    RWSS.allocate_memory();
    RWSS.load(path_binary); //??
    RWSS.write_PD_txt(path_binary, ".txt");
    return 0;
}
