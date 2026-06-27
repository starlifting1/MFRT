/* 
    ================================================================
    To read from .txt (simple particles data or whole particles data) 
    and write to .g1; or as versa.
    ===================================================================
*/

#ifndef _generate_gadget2_IC_
#define _generate_gadget2_IC_

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <omp.h>
#include "../diffu.h"
#include "../generate_samples.h"
#include "Gadget2FormatData_io.h"

int io_snapshot(int argc, char* argv[]);
int write_xvCartesian_to_binary_and_IC(
    string path_snapshot, const std::vector<PositionSpace>& pos, const std::vector<VelocitySpace>& vel, 
    float mass, int type, bool is_ID_from_1=true
);
int convert_gadget2_binary_to_txt(string path_binary);
int load_gadget2_binary_to_xvCartesian(string path_binary, std::vector<PositionSpace>& pos, std::vector<VelocitySpace>& vel, size_t N);

#endif