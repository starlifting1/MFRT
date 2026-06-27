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

# file_snapshot_like_a_galaxy=galaxy_general_NFW_1e4
# N_particls=10000
file_snapshot_like_a_galaxy=galaxy_general_NFW_1e6
N_particls=1000000

# file_snapshot_select=snapshot_000 #galaxy time 0.01 output
file_snapshot_select=snapshot_080 #near
# file_snapshot_select=snapshot_110 #merged



#### upload and download
home_server=/home/jygu
folder_MFRT_server=${home_server}/workroom/0prog/proj2/MFRT/

#: upload prog and data

scp -r ${folder_MFRT}diffu_r_simple_each/ jygu@192.168.70.42:${folder_MFRT_server}

# gadget2_server=${home_server}/workroom/0prog/GDDFAA_work/GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/
# galaxy_general_vacum=${HOME}/workroom/0prog/GDDFAA_work/GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/galaxy_general_vacum/
# scp -r ${galaxy_general_vacum} jygu@192.168.70.42:${gadget2_server}

# scp -r ${folder_MFRT}data_process/ jygu@192.168.70.42:${folder_MFRT_server}

# scp -r ${folder_MFRT}data/ jygu@192.168.70.42:${folder_MFRT_server}

# scp -r ${folder_MFRT}diffu_r_simple_each/compare.sh jygu@192.168.70.42:${folder_MFRT_server}diffu_r_simple_each/



#: download some results from server

# folder_diffu_simulated_server=${folder_MFRT_server}data/samples_simulated/
# scp -r jygu@192.168.70.42:${folder_diffu_simulated_server}snapshot_*Diffustatistics_read*.txt ${folder_diffu_simulated}

# scp -r jygu@192.168.70.42:${folder_MFRT_server}data/small/* ${folder_MFRT}data/small/



#### run
#######################################
#### ssh -x jygu@192.168.70.42
#### cd ~/workroom/0prog/GDDFAA_work/GDDFAA/step2_Nbody_simulation/gadget/Gadget-2.0.7/
#### mv galaxy_general/ galaxy_general_remerge_NFW_1e4/
#### mv galaxy_general/ galaxy_general_remerge_NFW_1e6/
#### cd ~/workroom/0prog/proj2/MFRT/diffu_r_simple_each/
#### 
#### nohup bash compare.sh &
#### jobs -l
#### ps -aux|grep run_big.param
#### ps -aux|grep out.exe
#### kill -9 [#the new id what [jobs -l] display]
#### cat nohup.out
#### mv nohup.out bak/
#######################################



set +e +u
echo -e "\nEnd."
