#!/usr/bin/env bash
#### to run
# python3 control ./out.exe IC
# move IC and run gadget2
# python3 control ./out.exe diffu
echo -e "Begin ... \n"
set -e -u



#### settings
folder_MFRT=${HOME}/workroom/0prog/proj2/MFRT/
folder_diffu_pos=${folder_MFRT}diffu_r_simple_each/
folder_diffu_vel=${folder_MFRT}diffu_v_simple_each/
folder_diffu_genIC=${folder_diffu_pos}generate_gadget2_IC/
folder_diffu_noised=${folder_diffu_pos}noised/
folder_diffu_simulated=${folder_diffu_pos}../data/samples_simulated/
folder_gadget=${HOME}/workroom/0prog/GDDFAA_work/GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/
folder_local_galaxy_general=galaxy_general/
folder_local_galaxy_vacum=galaxy_general_vacum/
folder_galaxy_general=${folder_gadget}${folder_local_galaxy_general}
file_galaxy_general=galaxy_general
folder_python_process=${folder_MFRT}data_process/
folder_saved_figs=${folder_MFRT}data/small/
file_snapshot_fractal=snapshot_fractal



#### 1. Using uniform or noised galaxy
# N_particles=10000
# N_particles=100000
N_particles=1000000
# N_particles=10000000
# N_particles=100000000
# N_particles=1000000000
N_iter=14
# N_iter=30

cd ${folder_diffu_pos}
#: uniform without noise
make; ./out.exe ${file_snapshot_fractal} ${N_particles} 137. 50. 1 1 0

#: big noise but all are less diffu
# make; 
./out.exe ${file_snapshot_fractal} ${N_particles} 137. 50. 1 1 ${N_iter}



# #### 2. Using merged galaxy
# # file_snapshot_like_a_galaxy=galaxy_general_NFW_1e4
# # N_particles=10000
# file_snapshot_like_a_galaxy=galaxy_general_NFW_1e6
# N_particles=1000000

# # file_snapshot_select=snapshot_000 #galaxy time 0.01 output
# # file_snapshot_select=snapshot_040 #near
# # file_snapshot_select=snapshot_080 #near
# file_snapshot_select=snapshot_120 #merged


# ## a. re-generate
# cd ${folder_diffu_pos}
# #() move file_snapshot_like_a_galaxy
# cd ${folder_diffu_genIC}
# make; ./generate_IC.exe ${file_galaxy_general} -1 137. 50. 1 1 ${file_snapshot_like_a_galaxy} 

# cd ${folder_python_process}
# python3 plot_simple.py ${folder_diffu_simulated}${file_galaxy_general}.g1.txt oldsnapshot
# echo -e "Plot xv, done.\n"
# cd ${folder_saved_figs} #to see figs

# ## b. simulate
# cd ${folder_gadget}
# if [ -d ${file_galaxy_general}/ ]; then
#     echo -e "There has been exist target name folder ${file_galaxy_general}. Backup it."
#     rm -rf ./${file_galaxy_general}_bak/ #delete the old bak
#     mv ./${file_galaxy_general}/ ./${file_galaxy_general}_bak/
# fi
# cp -r ${folder_local_galaxy_vacum} ${folder_local_galaxy_general}
# cd ${folder_local_galaxy_general}
# cp ${folder_diffu_simulated}${file_galaxy_general}.g1 ./
# cp ../Gadget2/Gadget2 ./

# echo -e "Running Gadget for simulation the galaxy ..."
# N_PROC=$(nproc)
# N_PROC=$((N_PROC/2))
# mpirun -np ${N_PROC} Gadget2 run_big.param

# ## c. diffu
# cd ${folder_diffu_genIC}
# cp ${folder_gadget}${folder_local_galaxy_general}snapshot/${file_snapshot_select} ${folder_diffu_simulated}
# # make; 
# ./generate_IC.exe ${file_galaxy_general} -1 137. 50. 1 0 ${file_snapshot_select} 
# cd ${folder_python_process}
# python3 plot_simple.py ${folder_diffu_simulated}${file_snapshot_select}.txt runsnapshot
# echo -e "Plot xv, done.\n"
# cd ${folder_saved_figs} #to see figs

# cd ${folder_diffu_pos}
# make; ./out.exe ${file_snapshot_select} ${N_particles} 137. 50. 0 1 0 



set +e +u
echo -e "\nEnd."
