#!/usr/bin/env python
# -*- coding:utf-8 -*-

# gm_name gm_num path_snapshot N_comp1 type_comp1 frac_mass_comp1 Mass_virial
# scale_length_comp1 scale_length_comp_median1 flatx_comp1, flaty_comp1, flatz_comp1
# fitmodel_name_xv powerS1 powerA1, powerB1, powerA2, powerB2 spin_L beta_sigma_total

# fitmodel_name_AA Jl_median, Jm_median, Jn_median Ol_median, Om_median, On_median
# J1_scale, J2_scale actions_coef_free_m, actions_coef_free_n
# actions_ratio_direct_m, actions_ratio_direct_n powerAA_E1 powerAA_A1, powerAA_B1, powerAA_A2, powerAA_B2

# ============================================================================================
# Description: A wrapper to fit galaxy models of mass density and action probability density.
# Author: Jianyu Gu
# ============================================================================================

import sys
import os
import re
# import pdb
# from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import analysis_data_distribution as add
import galaxy_models as gm
import change_params_galaxy_init as cpgi
import action_state_samples as asa
import transformation_some as ts
import KDTree_python as kdtp
import fit_galaxy_wrapper as fgw
import plot_galaxy_wrapper as pgw
import RW_data_CMGD as rdc
import action_error_by_near_time as sabnt
import triaxialize_galaxy as tg



####[] step0: set globally
filenme_IC = "../../step1_galaxy_IC_preprocess/step1_set_IC_DDFA/IC_param.txt"
mass1, N_comp1, v_sigma, cold_alpha, cold_alphamax, ls, seed = cpgi.read_IC_settings(filenme_IC)
# output_folder_name = "fit1/"
# os.makedirs("savefig/"+output_folder_name, exist_ok=True)
# output_folder_name = cpgi.read_output_folder_name()
# os.makedirs("savefig/"+output_folder_name, exist_ok=True)
output_folder_name = cpgi.read_output_folder_name_from_user_settings() #model1 name
# add.DEBUG_PRINT_V(0, output_folder_name)

model_ID = 0 #no use
M = mass1*N_comp1
N_ptcs = N_comp1 #int(1e4)
ls = ls
ds = M/(4./3*np.pi*ls**3)
Js = (gm.G*M*ls)**0.5
ar = np.array([1., 0.6, 0.3])

N_neighbour = 1024
bd1 = np.inf
# bd2 = 1e4
# bd2 = 5e4
# bd2 = 1e5
bd2 = 1e6
# bd2 = 1e8 #bd_fit
# is_show = True
is_show = False

tag_task = int(sys.argv[1])
# snapshot_ID = 0
# snapshot_ID = 5000
snapshot_ID = int(sys.argv[2])
name_MG = sys.argv[3]
suffix_action = ""
# suffix_action = ".variance"

folder_name = "../../step2_Nbody_simulation/gadget/Gadget-2.0.7/"
gm_name = "galaxy_general"
gm_names_inp = [""]
if name_MG!="default":
    gm_names_inp = re.split(r"[ ]+", name_MG)
    gm_name = gm_name+gm_names_inp[0]
# gm_name = "galaxy_general"
# gm_name = "galaxy_general_4_EinastoUsual_triaxial_soft5.0_count1e4"
suffix = ""
# suffix = ".SCF"
folder_many_params_fit = "../../step2_Nbody_simulation/gadget/paramters_fit/"



def main_step0_set_small():
    return 

def main_step1_set_MG(M, N_ptcs, ls, ds, ar):
    MG = fgw.Model_galaxy(M, ls, ds) #write MG
    Js = (gm.G*M*ls)**0.5
    # coef_boundary = 1.e2
    # coef_boundary = 1.e8
    coef_boundary = 1.e40
    scale_boundary = 1.e1
    scale_boundary = 1.e4
    # scale_boundary = 1.e2
    # power_boundary = 1.e2
    # power_boundary = 1.e3
    power_boundary = 1.e4
    axisratio_boundary = 1.e2
    MG.set_value("length_scale",    np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ls)
    MG.set_value("density_scale",   np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ds)
    MG.set_value("axis_ratio_x",    np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ar[0])
    MG.set_value("axis_ratio_y",    np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ar[1])
    MG.set_value("axis_ratio_z",    np.array([1., 1., 1./coef_boundary, 1.*coef_boundary])*ar[2])
    MG.set_value("rotate_angle_x",  np.array([0., 2*np.pi, 0., 2*np.pi])) # np.pi #divide by zero
    MG.set_value("rotate_angle_y",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("rotate_angle_z",  np.array([0., 2*np.pi, 0., 2*np.pi]))
    MG.set_value("action_scale",    np.array([Js, Js, Js/scale_boundary, Js*scale_boundary]))
    MG.set_value("scale_free_1",    np.array([Js, Js, Js/scale_boundary, Js*scale_boundary]))
    MG.set_value("scale_free_2",    np.array([Js, Js, Js/scale_boundary, Js*scale_boundary]))
    MG.set_value("scale_free_3",    np.array([Js/1000., Js/1000., 0., Js]))
    # MG.set_value("scale_free_3",    np.array([Js/100., Js/100., 0., Js/2.]))
    MG.set_value("scale_free_4",    np.array([Js, Js, Js/scale_boundary, Js*scale_boundary]))
    MG.set_value("log_penalty",     np.array([-10., -10., -100., 1.]))
    # MG.set_value("coef_total",      np.array([1., 1., 1.-0.1, 1.+0.1]))
    # MG.set_value("coef_total",      np.array([1., 1., 1.e-3, 1.e3]))
    MG.set_value("coef_total",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n3",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n2",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_n1",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_0",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_1",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_2",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_3",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_4",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_5",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_6",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_7",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_8",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_9",    np.array([1., 0., -coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p0",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p1",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p2",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p3",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p4",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p5",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p6",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p7",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p8",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_free_p9",    np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_1",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_2",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_exp_3",      np.array([1., 1., 1./coef_boundary, coef_boundary]))
    MG.set_value("coef_axis_1",     np.array([1., 1., 1./axisratio_boundary, axisratio_boundary]))
    MG.set_value("coef_axis_2",     np.array([1., 1., 1./axisratio_boundary, axisratio_boundary]))
    MG.set_value("coef_axis_3",     np.array([1., 1., 1./axisratio_boundary, axisratio_boundary]))
    MG.set_value("power_alpha",     np.array([1., 1., 0.,               power_boundary]))
    MG.set_value("power_beta",      np.array([3., 3., 0.,               power_boundary]))
    MG.set_value("power_total",     np.array([1., 1., 1./power_boundary/1e6,               power_boundary]))
    MG.set_value("power_free_1",    np.array([1., 1., 1./power_boundary/1e6,               power_boundary]))
    MG.set_value("power_free_2",    np.array([1., 1., 1./power_boundary/1e6,               power_boundary]))
    MG.set_value("power_free_3",    np.array([1., 1., 1./power_boundary/1e6,               power_boundary]))
    MG.set_value("power_free_4",    np.array([1., 1., 1./power_boundary/1e6,               power_boundary]))
    MG.set_value("power_Einasto",   np.array([1., 1., 1./power_boundary/1e6,               power_boundary]))
    # MG.set_value("power_Einasto",   np.array([1., 0.5, 1./power_boundary,               0.5])) #larger action is less

    return MG

def main_step2_DF(tag, snapshot_ID, is_read=False, is_grid=False):
    
    ## 0. values
    ps, mass = None, None
    tgts, DF = None, None
    xe, ye = None, None
    cl, meds = None, None
    DF_name = None

    galaxymodel_name = folder_name+gm_name+"/"
    cols = [0,1,2]
    pers = [0.5, 20., 50., 80., 99.5]

    ## 2.1. ps
    if int(tag)==1:
        path_file = galaxymodel_name+"txt/snapshot_%03d.txt"%(snapshot_ID)
        DF_name = "DF_x_mass"

        RG = rdc.Read_galaxy_data(path_file)
        # RG.data = RG.data[0:100] #cut data
        RG.AAAA_set_particle_variables(
            col_particle_x_coordinates=0, col_particle_v_velocities=3, 
            col_particle_IDs=7, col_particle_mass=8
        )

        x = RG.particle_x_coordinates # copy.deepcopy()
        v = RG.particle_v_velocities
        mass = RG.particle_mass
        N_dim = RG.system_space_dimension
        N_ptc = RG.system_particles_count
        add.DEBUG_PRINT_V(1, np.shape(x), np.shape(v))

        # vv:
        xv = np.hstack((x, v))
        bd = bd1
        xv_cl, cl, cln = add.screen_boundary_some_cols(xv, cols, -bd, bd, value_discard=bd*1e4)
        add.check_x_un_finite(xv, "xv after screen_boundary_some_cols")
        ps = xv #now: ps, mass, cl, meds

    ## 2.2. ps
    if int(tag)==2:
        DF_name = "DF_AA_one"
        path_file = galaxymodel_name+"aa/snapshot_%d.action.method_all%s.txt"%(snapshot_ID, suffix_action)
        
        RG = rdc.Read_galaxy_data(path_file)
        # RG.data = RG.data[0:100] #cut data
        RG.AAAA_set_particle_variables(
            col_particle_IDs=7-1, col_particle_mass=8-1 #??
        )

        data = RG.data
        mass = RG.particle_mass
        IDs = RG.particle_IDs
        Dim = gm.Dim #3
        iast = 28
        adur = 10
        AA_TF_FP = data[:, iast+adur*0:iast+adur*0+adur]
        AA_OD_FP = data[:, iast+adur*1:iast+adur*1+adur] #none
        AA_GF_FP = data[:, iast+adur*2:iast+adur*2+adur] #none
        iast += adur*5 # = 78
        AA_TF_DP = data[:, iast+adur*0:iast+adur*0+adur]
        AA_OD_DP = data[:, iast+adur*1:iast+adur*1+adur]
        AA_GF_DP = data[:, iast+adur*2:iast+adur*2+adur] #none

        # vv:
        AA_method = AA_TF_DP
        Act = AA_method[:, 0:3]
        Ang = AA_method[:, 3+1:7]
        Fre = AA_method[:, 7:10]
        add.DEBUG_PRINT_V(1, AA_TF_DP.shape, Act.shape, Fre.shape)
        AA = np.hstack((Act, Fre))
        bd = bd2
        AA_cl, cl, cln = add.screen_boundary_some_cols(AA, cols, 1e-2, bd, value_discard=bd*1e20)
        # AA_cl, cl, cln = add.screen_boundary_some_cols(AA, cols, 1./bd, bd, value_discard=bd*1e4)
        ps = AA #now: ps, mass, cl, meds    
        # add.DEBUG_PRINT_V(1, np.mean(data[:, 0]), np.median(np.abs(data[:, 0])), "mean and median of data[0]")
        # add.DEBUG_PRINT_V(0, np.mean(data[:, 3]), np.median(np.abs(data[:, 3])), "mean and median of data[0]")

        fne = galaxymodel_name+"aa/snapshot_%d.action.method_all%s.txt"%(snapshot_ID, ".variance")
        if os.path.exists(fne):
            rvae = np.loadtxt(fne)
            xe = rvae
        else:
            print("File `%s` do not exist. Set the default error be the same as the data."%(fne))
            xe = ps #before cl

    else:
        print("No such tag provided. None.")

    ## 3. DF
    if is_read:
        path_write = galaxymodel_name+"aa/snapshot_%d_%s.txt"%(snapshot_ID, DF_name)
        data_DF = np.loadtxt(path_write)
        tgts = data_DF[:,0:6] #[adjust] fit
        DF = data_DF[:,6]
    else:
        if not is_grid:
            ## [select]: samples: grid
            # tgts = (ps[:, cols])[cl] #note: used cl
            tgts = ps[cl] #note: used cl
            if xe is not None:
                xe = xe[cl]
            if ye is not None:
                ye = ye[cl]
        else:
            ## [select]: samples: grid
            N_grid = 100000
            # tag_func = 0 #[adjust] fit
            tag_func = 1
            # tag_func = 2
            # tag_func = 3
            tgts = add.generate_actions_grid_3d_3d(bd, N_grid, tag_func=tag_func)
            xe = tgts #??
            ye = DF

            ##: if nn point
            # from sklearn.neighbors import KDTree
            # tree_tgts = KDTree(AA_col)
            # distances, indices = tree_tgts.query(A3, k=1)
            # AA = AA_col[indices]
            ##: if knn Fre
            # [] SPH Fre by Act

        # density:
        KD = kdtp.KDTree_galaxy_particles(ps[:,cols], weight_extern_instinct=mass)
        # targets = [[0.,0.,0.], [1e2, 1e2, 1e2]]
        add.DEBUG_PRINT_V(1, np.shape(tgts), "tgts")
        DF = KD.density_SPH(tgts[:,cols]) #some are None #?? debug
        print("%s: "%(DF_name), DF) #now: tgts, DF

        # record: 
        path_write = galaxymodel_name+"aa/snapshot_%d_%s.txt"%(snapshot_ID, DF_name)
        data_write = np.hstack(( tgts, np.array([ DF ]).T ))
        RG.write_numpy_savetxt(path_write, data_write)

    ## 4. returned
    meds = [
        np.percentile(ps[cl,0], pers), 
        np.percentile(ps[cl,1], pers), 
        np.percentile(ps[cl,2], pers), 
        np.percentile(add.norm_l(ps[cl,0:3], axis=1), pers), 
        np.percentile(ps[cl,3], pers), 
        np.percentile(ps[cl,4], pers), 
        np.percentile(ps[cl,5], pers), 
        np.percentile(add.norm_l(ps[cl,3:6], axis=1), pers), 
        np.percentile(DF, pers)
    ]
    add.DEBUG_PRINT_V(1, np.shape(ps), np.shape(cl), np.shape(tgts), "shapes: ps, cl, tgts")
    print("medians of data ps and DF, tag=%d: "%(tag), meds)
    if int(tag)==2:
        ye = DF #??
    # exit(0)
    return ps, mass, tgts, DF, xe, ye, cl, meds

def main_step3_triaxialize(x, v, mass, snapshot_ID):
    # triaxialize:
    # boundary_rotate_refer = [60., 1000.] #kpc, km/s
    boundary_rotate_refer = [80., 1000.] #kpc, km/s
    # boundary_rotate_refer = None
    x, v, operators = tg.all_triaxial_process(
        x, v, mass, is_centralize_coordinate=True, 
        is_rotate_mainAxisDirection=True, boundary_rotate_refer=boundary_rotate_refer, 
        is_eliminate_totalRotation=False, is_by_DF=False, DF=None
    )

    # unpack the triaxialize parameters and write:
    x_mean_old, v_mean_old = operators[0][0], operators[0][1] #arrays 3 and 3 for translation
    T = operators[1][0] #array 3*3 for rotation
    OA = operators[2][0] #array 3 for elimination #only in IC before simulation

    galaxymodel_name = folder_name+gm_name+"/"
    path_file = galaxymodel_name+"aa/snapshot_%d_triaxialize.txt"%(snapshot_ID) #preprocess_snapshot
    
    file_handle = open(path_file, mode="w")
    file_handle.write("## operater values for actions under triaxial potential\n")
    file_handle.write("%e %e %e %e %e %e %s \n"%(x_mean_old[0], x_mean_old[1], x_mean_old[2], 
        v_mean_old[0], v_mean_old[1], v_mean_old[2], "#arrays 3 and 3 for translation centers of xv Cartesian"))
    file_handle.write("%e %e %e %e %e %e %e %e %e %s \n"%(T[0,0], T[0,1], T[0,2], 
        T[1,0], T[1,1], T[1,2], T[2,0], T[2,1], T[2,2], 
        "#array 3*3 for rotation main axis of interia moment"))
    file_handle.write("%e %e %e %s \n"%(OA[0], OA[1], OA[2], 
        "#array 3 for elimination total angular velocity #only in IC before simulation"))
    file_handle.close()
    print("Write `%s` ... Done."%(path_file))

    # write particles what SCF reads
    path_SCF_usual = folder_name+gm_name+"/"+"aa/galaxy_general.SCF.txt"
    tg.write_preprocessed_SCF(x, v, mass, path_file=path_SCF_usual)
    # add.DEBUG_PRINT_V(0, tg.centralize_coordinate(x, v, mass))
    return x, v, operators

def main_step4_fit_DF_x_mass(ps, mass, tgts, DF, xe, ye, cl, meds, MG):
    ####[] tables for params, with axisRatio, ds, rs, 
    # time power, common shape, nothing and gJ and colliSize
    
    #: processed data
    vinp, vout = np.abs(tgts), np.log10(DF) #xdata, ydata
    Js_data = meds[3][2]
    Js_expected = np.sqrt(gm.G*M*ls)
    axisratio_direct = add.axis_ratio_by_configuration_data(tgts)

    vinp_err, vout_err = xe, ye
    binsxy = pgw.plot_1d_DF_by_bins(vinp[:, 0:3], is_show=is_show)

    #: models
    combmodel = [
        [
            gm.AA_combination_sum, 
            "AA_combination_sum", 
            None
        ]
        # , 
        # [
        #     gm.AA_combination_freeCoef, 
        #     "AA_combination_freeCoef", 
        #     [
        #         "coef_free_p1", "coef_free_p2"
        #     ]
        # ]
    ]

    fitmodel = [
        [
            gm.DFXV_fCombinationFixed_DPL_log, #this function will be modifed to fix some params, the two axistatios
            "DFXV_fCombinationFixed_DPL_log", 
            [
                "power_free_1", "power_free_2", 
                "length_scale", "density_scale", 
                "log_penalty"
            ], 
            axisratio_direct[1:3] #the fixed params
        ]
        # , 
        # [
        #     gm.DFXV_fCombinationFree_DPL_log, 
        #     "DFXV_fCombinationFree_DPL_log", 
        #     [
        #         "coef_free_p1", "coef_free_p2", "power_free_1", 
        #         "power_free_2", "length_scale", "density_scale", 
        #         "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DFXV_fCombinationFree_MDPL_ExpPloy_log, 
        #     "DFXV_fCombinationFree_MDPL_ExpPloy_log", 
        #     [
        #         "power_free_1", "power_free_2", "power_free_3", "power_free_4", 
        #         "scale_free_1", "scale_free_2", "scale_free_3", 
        #         "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DFAA_fCombinationFree_MDPL_ExpPloy2, 
        #     "DFAA_fCombinationFree_MDPL_ExpPloy2", 
        #     [
        #         "power_free_1", "power_free_2", "power_free_3", "power_free_4", 
        #         "scale_free_1", "scale_free_2", "scale_free_3", 
        #         "log_penalty"
        #     ]
        # ]
    ]

    N_combmodel = len(combmodel)
    N_fitmodel = len(fitmodel)
    DL = [[0 for i in range(N_combmodel)] for j in range(N_fitmodel)]

    #: fit (here the combmodel is absorbed)
    for j in np.arange(N_fitmodel):
        WF = fgw.Wrap_func(MG, fitmodel[j][0], fitmodel[j][2], fitmodel[j][3]) #-log_panalty
        CF = fgw.Curve_fit_galaxy(WF, vinp, vout, vout_err, mf=5000)
        CF.run() #WF.funcfit, WF.reference_list
        CF_optimization, CF_covariance, CF_residual_sigma, CF_error_fit, CF_ymodel = CF.display()
        # #[adjust] fit
        # p_hand = [5.e-2,     1., 1.e-1, 1., 2.,  2.e5]
        # CF_ymodel = fitmodel[j][0](vinp, *p_hand)

        for i in np.arange(N_combmodel):
            print("fit and display: ", j,i)

            CB_fit = None
            if combmodel[i][2] is None:
                CB_fit = combmodel[i][0](vinp)
            else:
                N_combmodel_params = len(combmodel[i][2])
                combmodel_params = list(range(N_combmodel_params))
                for idx in np.arange(N_combmodel_params):
                    combmodel_params[idx] = MG.params_dict[combmodel[i][2][idx]][0] #this is after assign fitvalue to galaxy_model
                CB_fit = combmodel[i][0](vinp, *combmodel_params)
            
            # target1 = np.array([ CB_fit    ]).T
            target1 = np.array([ vout      ]).T
            KD1 = kdtp.KDTree_galaxy_particles(target1)
            dlocalm, dlocals = KD1.neighbour_average_and_scatter(target1, k=128)

            VA_data = vinp #[0]
            VB = CB_fit #[1]
            VC_data = vout #[2]
            VC_fit = CF_ymodel #[3]
            VD_mean = dlocalm #[4]
            VD_standard = dlocals #[5]
            VPFit = CF_optimization #[6]
            VOther = [CF_residual_sigma, CF_error_fit, 
                Js_data, Js_expected, 
                meds] #[7]
            V_direct = [axisratio_direct, Js_data, Js_expected, meds]

            DL[j][i] = [VA_data, VB, VC_data, VC_fit, 
                VD_mean, VD_standard, VPFit, VOther, V_direct]
        # end for i
    # end for j

    #: should unpack DL
    return DL, combmodel, fitmodel

def main_step5_fit_DF_AA_one(ps, mass, tgts, DF, xe, ye, cl, meds, MG):
    ####[] tables for params, with axisRatio, ds, rs, 
    # time power, common shape, nothing and gJ and colliSize
    
    #: processed data
    vinp, vout = tgts, np.log10(DF) #xdata, ydata
    AA_sum = np.sum(vinp[:, 0:3], axis=1)
    Js_data = meds[3][2]
    Js_expected = np.sqrt(gm.G*M*ls)
    # #: change weight
    # cl_sort, cl_newweight = add.change_weight_of_fitting_knn(AA_sum, 
    #     pers=[5., 95.], N_neighbour=100)
    # # cl_sort, cl_newweight = add.change_weight_of_fitting_knn(AA_sum, 
    # #     pers=[0.1, 99.9], N_neighbour=N_neighbour)
    # print("length of cl_newweight: ", len(cl_newweight)) #bd_fit
    # vinp = (vinp[cl_sort])[cl_newweight]
    # vout = (vout[cl_sort])[cl_newweight]

    vinp_err, vout_err = xe, np.log10(ye)
    binsxy = pgw.plot_1d_DF_by_bins(vinp[:, 0:3], is_show=is_show)
    # bd_pers = np.percentile(AA_sum, [20.,80.])
    # cl_between_pers = np.where((AA_sum>bd_pers[0])&(AA_sum<bd_pers[1]))[0]
    # vinp_err = vinp*0.01
    # vout_err = vout*0.01
    # vinp_err[cl_between_pers] = vinp[cl_between_pers]*0.2 #decrease weight of fitting
    # vout_err[cl_between_pers] = vout[cl_between_pers]*0.2 #decrease weight of fitting
    
    #: models
    combmodel = [
        # [
        #     gm.AA_combination_sumWeightFrequency_rateF1, 
        #     "AA_combination_sum", 
        #     None
        # ]
        # , 
        [
            gm.AA_combination_sum, 
            "AA_combination_sum", 
            None
        ]
        # , 
        # [
        #     gm.AA_combination_freeCoef, 
        #     "AA_combination_freeCoef", 
        #     [
        #         "coef_free_p1", "coef_free_p2"
        #     ]
        # ]
    ]
    
    fitmodel = [ # label_code_about_adjust_fit_function
        [
            gm.DFAA_fCombinationFreq_MPL_Expn_log, 
            "DFAA_fCombinationFreq_MPL_Expn_log",
            [
                # "power_free_1", "power_free_2", "power_free_3", 
                # "scale_free_1", "scale_free_2", "scale_free_3", 
                # "log_penalty"
                "power_free_1", "power_free_2", "power_free_4", 
                "scale_free_1", "scale_free_2", "scale_free_3", 
                "log_penalty"
            ]
        ]
        # , 
        # [ #version 2 used
        #     gm.DFAA_fCombinationFreq_DPL_Expn_log, 
        #     "DFAA_fCombinationFreq_DPL_Expn_log",
        #     [
        #         "power_free_1", "power_free_2", "power_free_3", "power_free_4", 
        #         "scale_free_1", "scale_free_2", "scale_free_3", 
        #         "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DFAA_fCombinationFree_MDPL_Expn_1_log, 
        #     "DFAA_fCombinationFree_MDPL_Expn_1_log",
        #     [
        #         "power_free_1", 
        #         "power_free_2", "power_free_3", 
        #         "coef_free_p0", 
        #         # "scale_free_2", 
        #         "power_Einasto", 
        #         "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DFAA_fCombinationFree_MDPL1_ExpPloy2, 
        #     "DFAA_fCombinationFree_MDPL1_ExpPloy2",
        #     [
        #         "power_free_1", "power_free_3", "power_free_4", 
        #         "scale_free_1", "scale_free_2", "scale_free_3", 
        #         "log_penalty"
        #     ]
        # ]
        # , 
        # [ #old version used: but not robust displayed in mc_fit
        #     gm.DFAA_fCombinationFree_MDPL_ExpPloy2, 
        #     "DFAA_fCombinationFree_MDPL_ExpPloy2",
        #     [
        #         "power_free_1", "power_free_2", "power_free_3", "power_free_4", 
        #         "scale_free_1", "scale_free_2", "scale_free_3", 
        #         "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DF_doubleGaussian_log, 
        #     "DF_doubleGaussian_log",
        #     [
        #         "coef_free_p1", "scale_free_1", "scale_free_3", 
        #         "coef_free_p2", "scale_free_2", "scale_free_4", 
        #         "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DFAA_MDPLE_log, 
        #     "DFAA_MDPLE_log",
        #     [
        #         "power_free_1", "power_free_2", "power_free_3", "power_free_4", 
        #         "power_Einasto", "coef_free_p1", 
        #         "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DFAA_fCombinationFree_debug_log, 
        #     "DFAA_fCombinationFree_debug_log",
        #     [
        #         "coef_free_p1", "coef_free_p2", 
        #         "power_free_1", "power_free_2", "power_free_3", "power_free_4", 
        #         "power_Einasto", "coef_free_p5", "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DF_simple_polymal, 
        #     "DF_simple_polymal",
        #     [
        #         "coef_free_0", "coef_free_1", "coef_free_2", "coef_free_3", 
        #         "coef_free_4", "coef_free_5", 
        #         "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DFAA_fCombinationFree_DPL_quadraticPL_Exp, 
        #     "DFAA_fCombinationFree_DPL_quadraticPL_Exp",
        #     [
        #         "coef_free_p1", "coef_free_p2", "coef_free_p3", 
        #         "power_free_1", "power_free_2", "power_free_3", 
        #         "scale_free_1", "scale_free_2", "scale_free_3", "log_penalty"
        #     ]
        # ]
        # , 
        # [
        #     gm.DFAA_fCombinationFree_modifiedDPLMultiplyEP_log,
        #     "DFAA_fCombinationFree_modifiedDPLMultiplyEP_log",
        #     [
        #         "coef_free_p1", "coef_free_p2", "coef_free_p3", "coef_free_p4", 
        #         "power_free_1", "power_free_2", "power_free_3", "power_free_4", 
        #         "power_Einasto", "coef_free_5", "log_penalty"
        #     ]
        # ]
    ]

    N_combmodel = len(combmodel)
    N_fitmodel = len(fitmodel)
    DL = [[0 for i in range(N_combmodel)] for j in range(N_fitmodel)]

    #: fit (here the combmodel is absorbed)
    for j in np.arange(N_fitmodel):
        WF = fgw.Wrap_func(MG, fitmodel[j][0], fitmodel[j][2]) #-log_panalty
        CF = fgw.Curve_fit_galaxy(WF, vinp, vout, vout_err, mf=5000)
        CF.run() #WF.funcfit, WF.reference_list
        CF_optimization, CF_covariance, CF_residual_sigma, CF_error_fit, CF_ymodel = CF.display()
        # #[adjust] fit
        # p_hand = [5.e-2,     1., 1.e-1, 1., 2.,  2.e5]
        # CF_ymodel = fitmodel[j][0](vinp, *p_hand)

        for i in np.arange(N_combmodel):
            print("fit and display: ", j,i)

            CB_fit = None
            if combmodel[i][2] is None:
                CB_fit = combmodel[i][0](vinp)
            else:
                N_combmodel_params = len(combmodel[i][2])
                combmodel_params = list(range(N_combmodel_params))
                for idx in np.arange(N_combmodel_params):
                    combmodel_params[idx] = MG.params_dict[combmodel[i][2][idx]][0] #this is after assign fitvalue to galaxy_model
                CB_fit = combmodel[i][0](vinp, *combmodel_params)
            
            # target1 = np.array([ CB_fit    ]).T
            target1 = np.array([ vout      ]).T
            KD1 = kdtp.KDTree_galaxy_particles(target1)
            dlocalm, dlocals = KD1.neighbour_average_and_scatter(target1, k=32)

            VA_data = vinp #[0]
            VB = CB_fit #[1]
            VC_data = vout #[2]
            VC_fit = CF_ymodel #[3]
            VD_mean = dlocalm #[4]
            VD_standard = dlocals #[5]
            VPFit = CF_optimization #[6]
            VOther = [CF_residual_sigma, CF_error_fit, 
                Js_data, Js_expected, 
                meds] #[7]
            # add.DEBUG_PRINT_V(1, Js_data, Js_expected)

            DL[j][i] = [VA_data, VB, VC_data, VC_fit, 
                VD_mean, VD_standard, VPFit, VOther]
        # end for i
    # end for j

    #: should unpack DL
    return DL, combmodel, fitmodel

def main_step5_fit_mc_DF_AA_one(ps, mass, tgts, DF, xe, ye, cl, meds, MG, info=None, special_init_guess=None):
    ####[] tables for params, with axisRatio, ds, rs, 
    # time power, common shape, nothing and gJ and colliSize
    
    #: processed data
    # vinp, vout = tgts[0:1000], np.log10(DF[0:1000]) #xdata, ydata
    vinp, vout = tgts, np.log10(DF) #xdata, ydata
    AA_sum = np.sum(vinp[:, 0:3], axis=1)
    Js_data = meds[3][2]
    Js_expected = np.sqrt(gm.G*M*ls)

    vinp_err, vout_err = xe, ye
    binsxy = pgw.plot_1d_DF_by_bins(vinp[:, 0:3], is_show=is_show)
    #: models
    combmodel = [
        [
            gm.AA_combination_sum, 
            "AA_combination_sum", 
            None
        ]
    ]

    fitmodel = [ #[adjust] fit
        [ #now used
            gm.DFAA_fCombinationFree_MDPL_ExpPloy2, 
            "DFAA_fCombinationFree_MDPL_ExpPloy2",
            [
                "power_free_1", "power_free_2", "power_free_3", "power_free_4", 
                "scale_free_1", "scale_free_2", "scale_free_3", 
                "log_penalty"
            ]
        ]
    ]

    N_combmodel = len(combmodel)
    N_fitmodel = len(fitmodel)
    DL = [[0 for i in range(N_combmodel)] for j in range(N_fitmodel)]

    #: fit (here the combmodel is absorbed)
    for j in np.arange(N_fitmodel):
        # WF = fgw.Wrap_func(MG, fitmodel[j][0], fitmodel[j][2]) #-log_panalty
        # CF = fgw.Curve_fit_galaxy(WF, vinp, vout, vout_err, mf=5000)
        # CF.run() #WF.funcfit, WF.reference_list
        # CF_optimization, CF_covariance, CF_residual_sigma, CF_error_fit, CF_ymodel = CF.display()
        
        # vinp, vout = vinp[0:200], vout[0:200]
        # if special_init_guess is not None:
        #     N_pfmj = len(fitmodel[j][2])
        #     for ip in np.arange(N_pfmj):
        #         MG[fitmodel[j][2][ip]][1] = special_init_guess[ip]
        WF = fgw.Wrap_func(MG, fitmodel[j][0], fitmodel[j][2]) #-log_panalty
        if special_init_guess is not None:
            WF.reference_list = special_init_guess
            WF.min_list = special_init_guess/1.2
            WF.max_list = special_init_guess*1.2
        # n_step = 20
        n_step = 4000
        MC = fgw.MCMC_fit_galaxy(WF, vinp, vout, vout_err, n_step=n_step)
        MC.run() #burn in ??
        name_corner = ""
        if info is not None:
            name_corner = info[0]
        CF_optimization, CF_covariance, CF_residual_sigma, CF_error_fit, CF_ymodel = MC.display(name=name_corner)
        
        for i in np.arange(N_combmodel):
            print("fit and display: ", j,i)

            CB_fit = None
            if combmodel[i][2] is None:
                CB_fit = combmodel[i][0](vinp)
            else:
                N_combmodel_params = len(combmodel[i][2])
                combmodel_params = list(range(N_combmodel_params))
                for idx in np.arange(N_combmodel_params):
                    combmodel_params[idx] = MG.params_dict[combmodel[i][2][idx]][0] #this is after assign fitvalue to galaxy_model
                CB_fit = combmodel[i][0](vinp, *combmodel_params)
            
            # target1 = np.array([ CB_fit    ]).T
            target1 = np.array([ vout      ]).T
            KD1 = kdtp.KDTree_galaxy_particles(target1)
            dlocalm, dlocals = KD1.neighbour_average_and_scatter(target1, k=32)

            VA_data = vinp #[0]
            VB = CB_fit #[1]
            VC_data = vout #[2]
            VC_fit = CF_ymodel #[3]
            VD_mean = dlocalm #[4]
            VD_standard = dlocals #[5]
            VPFit = CF_optimization #[6]
            VOther = [CF_residual_sigma, CF_error_fit, 
                Js_data, Js_expected, meds] #[7]
            # add.DEBUG_PRINT_V(1, Js_data, Js_expected)

            DL[j][i] = [VA_data, VB, VC_data, VC_fit, 
                VD_mean, VD_standard, VPFit, VOther]
        # end for i
    # end for j

    #: should unpack DL
    return DL, combmodel, fitmodel

def main_step6_record_and_plot_fitvalues(MG, DL, combmodel, fitmodel, tag=2):
    ##[] unpack DL
    galaxy_basic_info = [
        MG.params_dict["mass"][1], #total mass
        N_ptcs, #total particle
        MG.params_dict["length_scale"][1], 
        MG.params_dict["density_scale"][1], 
        MG.params_dict["action_scale"][1]
    ]
    galaxy_fit_params_to_compare = DL[0][0][6]
    print("galaxy_basic_info: ", galaxy_basic_info)
    print("galaxy_basic_info: debug: ", MG.params_dict["power_Einasto"][0])
    print("galaxy_fit_params_to_compare: ", galaxy_fit_params_to_compare)

    N_combmodel = len(combmodel)
    N_fitmodel = len(fitmodel)
    N_pf = len(galaxy_fit_params_to_compare)
    Js_data = DL[0][0][7][2]
    Js_expected = DL[0][0][7][3]
    meds = DL[0][0][7][4]
    print("scale_action by data: ", Js_data)
    print("scale_action by expected: ", Js_expected)

    ##[] record one (select DL[j=0][i=0]) #preprocess_snapshot
    galaxymodel_name = folder_name+gm_name+"/"
    galaxy_params_fit = galaxymodel_name+"aa/"+"snapshot_%d_DF_params_fit"%(snapshot_ID)
    
    #: tag0
    galaxy_params_settings = galaxy_params_fit+".MG.txt"
    file_handle = open(galaxy_params_settings, mode="w")
    # file_handle.write("## settings \n")
    file_handle.write("%d # %s \n"%(0, "model_ID"))
    file_handle.write("%e # %s \n"%(M, "total mass"))
    file_handle.write("%e # %s \n"%(N_ptcs, "total count of particles (comp1)"))
    file_handle.write("%e # %s \n"%(ls, "expected scale length"))
    file_handle.write("%e # %s \n"%(ds, "expected scale density"))
    file_handle.write("%e # %s \n"%(Js_expected, "expected scale action"))
    file_handle.write("%e # %s \n"%(ar[0], "expected axis_ratio_x"))
    file_handle.write("%e # %s \n"%(ar[1], "expected axis_ratio_y"))
    file_handle.write("%e # %s \n"%(ar[2], "expected axis_ratio_z"))
    file_handle.close()
    
    #: tag1 or tag2
    if tag==1:
        galaxy_params_fit += ".xv.txt"
    elif tag==2:
        galaxy_params_fit += ".AA.txt"
    else:
        print("tag: ", tag)
        print("No such tag provided. Exit.")
        exit(0)
    file_handle = open(galaxy_params_fit, mode="w")
    # file_handle.write("## paramters by %s \n"%(fitmodel[0][1]))
    file_handle.write("%d # %s \n"%(0, "model_ID"))
    for k in np.arange(N_pf):
        file_handle.write("%e # %s \n"%(galaxy_fit_params_to_compare[k], fitmodel[0][2][k]))
    #: fit params of simple DPL: "coef_free_p1", "coef_free_p2", "power_free_1", "power_free_2", "density_scale"
    file_handle.write("%e # %s \n"%(meds[0][2], "scale: median of x or actions, coor1"))
    file_handle.write("%e # %s \n"%(meds[1][2], "scale: median of x or actions, coor2"))
    file_handle.write("%e # %s \n"%(meds[2][2], "scale: median of x or actions, coor3"))
    file_handle.write("%e # %s \n"%(meds[3][2], "scale: median of x or actions, l2_norm"))
    file_handle.write("%e # %s \n"%(meds[4][2], "scale: median of v or frequencies, coor1"))
    file_handle.write("%e # %s \n"%(meds[5][2], "scale: median of v or frequencies, coor2"))
    file_handle.write("%e # %s \n"%(meds[6][2], "scale: median of v or frequencies, coor3"))
    file_handle.write("%e # %s \n"%(meds[7][2], "scale: median of v or frequencies, l2_norm"))
    file_handle.close()
    print("Write parameters of a galaxy ... Done.")

    ##[] plot one
    # Js = Js_data
    # print("the selected plot-scale_action is by data: ", Js)
    Js = Js_expected
    print("the selected plot-scale_action is by expected: ", Js)
    for j in np.arange(N_fitmodel):
        dp_fit = [0 for i in range(N_combmodel)]
        dp_err = [0 for i in range(N_combmodel)]
        dp_local_mean_var0 = [0 for i in range(N_combmodel)]
        dp_local_scatter_var0 = [0 for i in range(N_combmodel)]
        percentiles = None
        for i in np.arange(N_combmodel):
            print("plot: ", j,i)
            # Jcombine = DL[j][i][1]/Js
            Jcombine = DL[j][i][1]
            percentiles = np.percentile(Jcombine, [0.5, 99.5])
            dp_fit[i] = [
                [Jcombine, DL[j][i][2], "VB_%s versus VC_data"%(combmodel[i][1]), 0], #k=0
                [Jcombine, DL[j][i][3], "VB_%s versus VC_fit"%(combmodel[i][1]), 0]  #k=1
            ]
            dp_err[i] = [
                [Jcombine, DL[j][i][7][0], 
                    "VB_%s versus VC_residual_sigma"%(combmodel[i][1]), 0]
            ]
            dp_local_mean_var0[i] = [
                [Jcombine, DL[j][i][2], "VB_%s versus VC_data"%(combmodel[i][1]), 0], 
                [Jcombine, DL[j][i][4], "VB_localAverage_%s versus VC_data"%(combmodel[i][1]), 0]
            ]
            dp_local_scatter_var0[i] = [
                [Jcombine, DL[j][i][5], 
                    "VB_localScatter_%s versus VC_data"%(combmodel[i][1]), 0]  #k=0
            ]
        # end for i

        #: plot fit paramters:
        PLOT = pgw.Plot_model_fit()
        datapack = [dp_fit, dp_err, dp_local_mean_var0, dp_local_scatter_var0] #the plot input is this datapack
        xl = [
            "scaled combination_actions", 
            "scaled combination_actions", 
            "scaled combination_actions localAverage", 
            "scaled combination_actions localScatter"
        ]
        yl = [
            "log10 DF/1.", "residual of log10 DF/1.", 
            "log10 DF/1.", "log10 DF/1."
        ]
        sn = output_folder_name+"/"+"fitmodel_%d_"%(j)+"DFAA_"+fitmodel[j][1]+"_versus_combmodel_plot"\
            +"_of_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
        ep = str(Js)
        tx = str(galaxy_fit_params_to_compare)
        st = sn+"\ninfo = "+ep+"\nthe fitvalue: "+tx
        lim = None
        lim = [[0.,0.],[-20.,-2.]] #None
        PLOT.plot_actions_Comb_NDF_subplot(
            datapack, xl=xl, yl=yl, lim=lim, bd_much=percentiles, 
            suppertitle=st, savename=sn, is_show=is_show
        )
    # end for j

    PLOT = pgw.Plot_model_fit()
    # mcmc and plot #??
    # PLOT.plot_x_scatter3d_general() #??

    return tag

def plot_comb_and_DF_in_2d_and_3d(func, fitvalue, xdata, ydata, tag_comb=0):

    #[] 1-1, 2d, value
    xcomb = None #xdata is JO (actions and frequencies), whose shape is (N,6)
    if tag_comb==0:
        xcomb = gm.AA_combination_sum(xdata)
    elif tag_comb==1:
        xcomb = gm.AA_combination_sumWeightFrequency_rateF1(xdata)
    else:
        xcomb = gm.AA_combination_freeCoef(xdata, tag_comb[0], tag_comb[1])
    yfit = func(xdata, *fitvalue)

    f_list = None
    is_save = True
    PLOT = pgw.Plot_model_fit()
    
    dim = 2
    plot_list = [
        [xcomb, ydata], 
        [xcomb, yfit]
    ]
    label_list = [
        "NDF_data", 
        "NDF_fit"
    ]

    xyzlim = [
        [1., bd2], 
        None
        # [-20., -2.]
    ]
    scalevalue = [
        Js, 
        None
    ]
    xyztitle = [
        "lincomb_actions (log10 unit)", 
        "NDF (log10 unit)"
    ]
    xyzlogscale = [
        True, 
        False
    ]

    pathname = output_folder_name
    figurename = "value_of_data_and_fit_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
    textname = "scale: "+str(Js)+"\nfitvalue: "+str(fitvalue)
    
    PLOT.plot_scatter_2d_or_3d_with_color(
        plot_list, f_list=f_list, label_list=label_list, 
        dim=dim, xyzlogscale=xyzlogscale, xyzlim=xyzlim, xyztitle=xyztitle, scalevalue=scalevalue, 
        pathname=pathname, figurename=figurename, textname=textname, is_save=is_save, is_show=is_show
    )

    #[] 1-2, 2d, error
    # y_relative_error = yfit/ydata-1.
    y_relative_error = yfit-ydata
    plot_list = [
        [xcomb, y_relative_error]
    ]
    label_list = [
        "NDF_relative_error"
    ]

    figurename = "relative_error_of_data_and_fit_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
    
    PLOT.plot_scatter_2d_or_3d_with_color(
        plot_list, f_list=f_list, label_list=label_list, 
        dim=dim, xyzlogscale=xyzlogscale, xyzlim=xyzlim, xyztitle=xyztitle, scalevalue=scalevalue, 
        pathname=pathname, figurename=figurename, textname=textname, is_save=is_save, is_show=is_show
    )

    #[] 3d
    if 0:
        J_l = xdata[:,0]
        J_mn = xcomb-J_l
        yfit = func(xdata, *fitvalue)
        # add.DEBUG_PRINT_V(0, np.min(yfit), np.max(yfit))

        dim = 3
        plot_list = [
            [J_l, J_mn, ydata], 
            [J_l, J_mn, yfit]
        ]
        label_list = [
            "NDF_data", 
            "NDF_fit"
        ]

        xyzlim = [
            [1./bd2*1e2, bd2], 
            [1./bd2*1e2, bd2], 
            None
            # [-20., -2.]
        ]
        scalevalue = [
            Js, 
            Js, 
            None
        ]
        xyztitle = [
            "lincomb_actions_l (log10 unit)", 
            "lincomb_actions_mn (log10 unit)", 
            "NDF (log10 unit)"
        ]
        xyzlogscale = [
            True, 
            True, 
            False
        ]

        pathname = output_folder_name
        figurename = "value_of_data_and_fit_3d_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
        textname = "scale: "+str(Js)+"\nfitvalue: "+str(fitvalue)
        
        PLOT.plot_scatter_2d_or_3d_with_color(
            plot_list, f_list=f_list, label_list=label_list, 
            dim=dim, xyzlogscale=xyzlogscale, xyzlim=xyzlim, xyztitle=xyztitle, scalevalue=scalevalue, 
            pathname=pathname, figurename=figurename, textname=textname, is_save=is_save
        )

        #[] 2-2, 3d, error
        y_relative_error = yfit/ydata-1.
        plot_list = [
            [J_l, J_mn, y_relative_error]
        ]
        label_list = [
            "NDF_relative_error"
        ]
        xyzlim = [
            [1./bd2*1e2, bd2], 
            [1./bd2*1e2, bd2], 
            None
            # [-20., -2.]
        ]
        
        figurename = "relative_error_of_data_and_fit_3d_"+gm_name+"_snapshot_%d"%(snapshot_ID)+suffix
        
        PLOT.plot_scatter_2d_or_3d_with_color(
            plot_list, f_list=f_list, label_list=label_list, 
            dim=dim, xyzlogscale=xyzlogscale, xyzlim=xyzlim, xyztitle=xyztitle, scalevalue=scalevalue, 
            pathname=pathname, figurename=figurename, textname=textname, is_save=is_save
        )

    return 0

def main_step7_write_info_and_params(gm_names):
    ####[] write galaxy info, params about xv and params about AA 
    # and write multi models and compare
    # If you want to compare snapshots of one model only, just let len(gm_names) be 1
    print("The models names are:")
    for gn in gm_names:
        print(gn)
    print("by user input.")
    
    ##[] read
    params0 = []
    params1 = []
    params2 = []
    name_p0 = None
    name_p1 = None
    name_p2 = None
    idx = 0
    for gn in gm_names:
        galaxymodel_name_folder = ""
        if gn=="" or gn==None:
            galaxymodel_name_folder = folder_name+"galaxy_general"+"/"+"aa/"
        else:
            galaxymodel_name_folder = folder_name+"galaxy_general"+"_"+gn+"/"+"aa/"
        name_ld = os.listdir(galaxymodel_name_folder)
        name_ld.sort()
        read_did = [False, False, False]
        for name_file in name_ld:
            if name_file.find("_DF_params_fit.MG.txt")!=-1:
                pm = np.loadtxt(galaxymodel_name_folder+name_file)
                params0.append(pm)
                if read_did[0]==False:
                    name_p0 = add.read_single_comment_of_each_line(
                        galaxymodel_name_folder+name_file, comment_symbol="#")
                    read_did[0] = True
            if name_file.find("_DF_params_fit.xv.txt")!=-1 and name_file!="example_snapshot_DF_params_fit.xv.txt":
                pm = np.loadtxt(galaxymodel_name_folder+name_file)
                params1.append(pm)
                if read_did[1]==False:
                    name_p1 = add.read_single_comment_of_each_line(
                        galaxymodel_name_folder+name_file, comment_symbol="#")
                    read_did[1] = True
            if name_file.find("_DF_params_fit.AA.txt")!=-1:
                pm = np.loadtxt(galaxymodel_name_folder+name_file)
                params2.append(pm)
                if read_did[2]==False:
                    name_p2 = add.read_single_comment_of_each_line(
                        galaxymodel_name_folder+name_file, comment_symbol="#")
                    read_did[2] = True
            idx += 1
    N_ss = len(params0)
    if not( len(params0)==len(params1) and len(params0)==len(params2) ):
        print("len of MGs: ", len(params0))
        print("len of XVs: ", len(params1))
        print("len of AAs: ", len(params2))
        print("Error when running: The counts of galaxies in type \".MG\", \".xv\" and \".AA\" is not same. Exit.")
        exit(0)
    N_pp0 = len(params0[0])
    N_pp1 = len(params1[0])
    N_pp2 = len(params2[0])
    add.DEBUG_PRINT_V(1, [N_pp0, N_pp1, N_pp2], N_ss, "counts of multi MG")

    ##[] write
    params0_record = np.zeros((N_pp0, N_ss))
    params1_record = np.zeros((N_pp1, N_ss))
    params2_record = np.zeros((N_pp2, N_ss))
    for j in np.arange(N_ss):
        for i in np.arange(N_pp0):
            params0_record[i,j] = params0[j][i]
        for i in np.arange(N_pp1):
            params1_record[i,j] = params1[j][i]
        for i in np.arange(N_pp2):
            params2_record[i,j] = params2[j][i]

    path_params0 = folder_many_params_fit+"many_MG_params_fit.txt"
    np.savetxt(path_params0, params0_record)
    fh = open(path_params0, mode="r")
    cs = fh.readlines()
    fh.close()
    for i in np.arange(N_pp0):
        cs[i] = cs[i][:-1]+name_p0[i]+cs[i][-1] #cs[-1] is "\n"
    fh = open(path_params0, mode="w")
    fh.writelines(cs)
    fh.close()
    path_params1 = folder_many_params_fit+"many_xv_params_fit.txt"
    np.savetxt(path_params1, params1_record)
    fh = open(path_params1, mode="r")
    cs = fh.readlines()
    fh.close()
    for i in np.arange(N_pp1):
        cs[i] = cs[i][:-1]+name_p1[i]+cs[i][-1] #cs[-1] is "\n"
    fh = open(path_params1, mode="w")
    fh.writelines(cs)
    fh.close()
    path_params2 = folder_many_params_fit+"many_AA_params_fit.txt"
    np.savetxt(path_params2, params2_record)
    fh = open(path_params2, mode="r")
    cs = fh.readlines()
    fh.close()
    for i in np.arange(N_pp2):
        cs[i] = cs[i][:-1]+name_p2[i]+cs[i][-1] #cs[-1] is "\n"
    fh = open(path_params2, mode="w")
    fh.writelines(cs)
    fh.close()
    print("Write parameters of multi galaxy ... Done.")

    return 0

def main_step8_compare_DFAA_parameters_of_multi_galaxies():
    ####[] plot and compare
    path_params0 = folder_many_params_fit+"many_MG_params_fit.txt"
    params0_record = np.loadtxt(path_params0)
    name_p0 = add.read_single_comment_of_each_line(path_params0, comment_symbol="#")
    path_params1 = folder_many_params_fit+"many_xv_params_fit.txt"
    params1_record = np.loadtxt(path_params1)
    name_p1 = add.read_single_comment_of_each_line(path_params1, comment_symbol="#")
    path_params2 = folder_many_params_fit+"many_AA_params_fit.txt"
    params2_record = np.loadtxt(path_params2)
    name_p2 = add.read_single_comment_of_each_line(path_params2, comment_symbol="#")
    if not (len(params0_record)==len(name_p0) and len(params1_record)==len(name_p1) 
        and len(params2_record)==len(name_p2)):
        print("There something wrong about lines in the file reading. Exit.")
        exit(0)
    prl = [params0_record, params1_record, params2_record]
    pcl = [name_p0, name_p1, name_p2]
    snl = ["_MG", "_xv", "_AA"]
    add.DEBUG_PRINT_V(1, np.shape(prl[0]), np.shape(prl[1]), np.shape(prl[2]), "shape of prl")

    ##: plot each fit by reading the fit params
    # fitmodel, combmodel

    ##: plot multi params
    for j in np.arange(3):
        pgw.plot_parameters_mg_list(prl[j], label_list=pcl[j], sn=snl[j], is_show=is_show)
    return 0

def read_and_fits_param_to_compare(MG, DL, combmodel, fitmodel, tag=1):
    #work11_paramsresult
    ## read xv
    ## fit DPL and reuse axisratio, spin and powerlaw
    ## read AA
    ## fit DFA and check too bad #??
    ## plot two by time, space and param
    ## server and tell space #??
    ## FFPE theoy prog paper
    return 0



# main()
if __name__ == '__main__':
    
    ####[] settings
    # [] size (AA_combination_sumWeightFrequency 
    # "np.sum(J*O,axis=1)" (997004,3) (997004,0)), 
    # KDTree leaf mean (less fit points), 
    # plot and see DF, 
    # run fit, write params of each model, 
    # more IC running and more models, 
    # server, debug PPOD, background, paper

    ##[] step0 small
    main_step0_set_small() #Nothing

    ##[] step1 MG
    MG = main_step1_set_MG(M, N_ptcs, ls, ds, ar)



    #### record a series of snapshots
    if tag_task==5:
        ps, mass, tgts, DF, xe, ye, cl, meds = main_step2_DF(1, snapshot_ID)
        x, v = ps[:,0:3], ps[:,3:6] #note: without cl

        x, v, operators = main_step3_triaxialize(x, v, mass, snapshot_ID)
        ps = np.hstack((x,v)) #new xv

        DL, combmodel, fitmodel = main_step4_fit_DF_x_mass(ps, mass, tgts, DF, xe, ye, cl, meds, MG)
        # main_step6_record_and_plot_fitvalues(MG, DL, combmodel, fitmodel, tag=1)
        read_and_fits_param_to_compare(MG, DL, combmodel, fitmodel, tag=1)



    #### XV
    if tag_task==1:
        ##[] step2 xv data
        ps, mass, tgts, DF, xe, ye, cl, meds = main_step2_DF(1, snapshot_ID)
        x, v = ps[:,0:3], ps[:,3:6] #note: without cl

        ##[] step3 triaxialize and write info
        x, v, operators = main_step3_triaxialize(x, v, mass, snapshot_ID)
        ps = np.hstack((x,v)) #new xv

        # ##[] step4 fit DF_x_mass
        # DL, combmodel, fitmodel = main_step4_fit_DF_x_mass(ps, mass, tgts, DF, xe, ye, cl, meds, MG)
        # main_step6_record_and_plot_fitvalues(MG, DL, combmodel, fitmodel, tag=1)


        
    #### AA
    elif tag_task==2:
        ##[] calculate action
        #() TACT

        ##[] step5 #[adjust] fit
        ps, mass, tgts, DF, xe, ye, cl, meds = main_step2_DF(2, snapshot_ID, is_read=False, is_grid=False) #bd_fit

        ##[] step6 fit DF_AA_one
        DL, combmodel, fitmodel = main_step5_fit_DF_AA_one(ps, mass, tgts, DF, xe, ye, cl, meds, MG)
        
        # special_init_guess = DL[0][0][6] #cf opt
        # info = [output_folder_name+"/"+"snapshot_%d"%(snapshot_ID)]
        # DL, combmodel, fitmodel = main_step5_fit_mc_DF_AA_one(ps, mass, tgts, DF, xe, ye, cl, meds, MG, info=info, special_init_guess=special_init_guess)
        
        main_step6_record_and_plot_fitvalues(MG, DL, combmodel, fitmodel, tag=2)



    #### compare
    elif tag_task==3:
        ##[] step8 plot compare DF
        #! loop the upper steps with different IC
        main_step7_write_info_and_params(gm_names_inp)

        ##[] step11 plot multi
        main_step8_compare_DFAA_parameters_of_multi_galaxies()

    #### debug
    elif tag_task==4:
        # func = gm.DFAA_fCombinationFree_MDPL_ExpPloy2
        # #[] this fit function is sensitive and unstable to fit value, so curve_fit is not stable and mcmc has bad mean
        # #[??] too much power indexed and wrong Gauss mean
        # #[??] mean err and foci debug, and run
        # # fitvalue = np.array([
        # #     7.649e2, 2.034e-3, 7.405e2, 2.037e-3, 
        # #     3.250e4, 3.053e5, 4.726e5
        # # ]) #curve_fit 90
        # fitvalue = np.array([
        #     7.215e2, 2.197e-3, 7.827e2, 1.904e-3, 
        #     3.558e4, 2.853e5, 4.505e5
        # ]) #emcee 90
        # # tgts, DF = (np.random.random((1000, 6))+0.1)*1e4, (np.random.random(1000)+0.1)*(1e-6)
        
        ps, mass, tgts, DF, xe, ye, cl, meds = main_step2_DF(2, snapshot_ID, is_read=False, is_grid=False)
        # tgts = tgts*(1. + np.random.random(np.shape(tgts))*0.1)
        # DF = DF*(1. + np.random.random(np.shape(DF))*0.1)
        # xdata = tgts[0:10000]
        # ydata = np.log10(DF)[0:10000]
        print(tgts.shape, DF.shape, xe.shape, ye.shape)
        Jl = tgts[:,0]
        Jm = tgts[:,1]
        Jn = tgts[:,2]
        JmAndJn = tgts[:,1]+tgts[:,2]
        fJ = np.log10(DF)
        plot_dim = 3
        datapack = [[Jl, JmAndJn, fJ, "label"]]
        bd2p = bd2 #bd2
        xinfo = [[0.,bd2p], None, "Jl ()"]
        # yinfo = [[0.,bd2], None, "JmAndJn ()"]
        # zinfo = [None, None, "fJ ()"]
        yinfo = [[0.,bd2p], None, "Jm ()"]
        zinfo = [[0.,bd2p], None, "Jn ()"]
        nameinfo = ["./", "~~", "sometexts"]

        PLOT = pgw.Plot_model_fit()
        dim = 3
        plot_list = [
            # [Jl, Jm, Jn]
            [np.log10(Jl), np.log10(Jm), np.log10(Jn)]
        ]
        f_list = [fJ]
        label_list = [
            "action_space_points"
        ]
        xyzlim = [
            # [1./bd2p*1e2, bd2p], 
            # [1./bd2p*1e2, bd2p], 
            # [1./bd2p*1e2, bd2p]
            None, 
            None, 
            None
        ]
        # scalevalue = [
        #     Js, 
        #     Js, 
        #     Js
        # ]
        xyztitle = [
            "J_lambda (log10 unit)", 
            "J_mu (log10 unit)", 
            "J_nv (log10 unit)"
        ]
        # xyzlogscale = [
        #     False, 
        #     False, 
        #     False
        # ]
        PLOT.plot_scatter_2d_or_3d_with_color(plot_list, f_list=f_list, 
            label_list=label_list, dim=dim, xyzlim=xyzlim, xyztitle=xyztitle
        )

        # pgw.plot_corse_grainedly_data_from_file(datapack, plot_dim, xinfo, yinfo, zinfo, nameinfo=nameinfo)
        # exit(0)

        # # label_code_about_adjust_fit_function
        # DL, combmodel, fitmodel = main_step5_fit_DF_AA_one(ps, mass, tgts, DF, xe, ye, cl, meds, MG)
        # fitvalue = DL[0][0][6] #cf opt
        # func = fitmodel[0][0]
        # plot_comb_and_DF_in_2d_and_3d(func, fitvalue, tgts, np.log10(DF), tag_comb=0)

    else:
        print("tag_task: ", tag_task)
        print("No such tag_task provided. Exit.")
        exit(0)
