#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import warnings
import pandas as pd

import analysis_data_distribution as ads
import diffusion_coef_integration as dci
import diffu_basic as dbc
import observed_data_process as odp
import generate_fractal_sample as gfs
import diffu_calling_cpp as dcc


PERLIN_MODIFIED_POSITION_TAGS = {"noised"}
FRACTAL_DIMENSION_INDEX_PATH = "../data/fractal_dimension_by_figure.txt"


def append_fractal_dimension_record(
    records, figure_relative_path, snapshot_name, N_particles, Dim_frac,
    displayed_on_figure, figure_kind, perlin_modified
):
    if records is None:
        return
    sample_type = (
        snapshot_name[len("type_"):]
        if snapshot_name.startswith("type_") else snapshot_name
    )
    records.append({
        "figure_relative_path": figure_relative_path,
        "sample_type": sample_type,
        "N": N_particles,
        "fitted_D_frac": Dim_frac,
        "displayed_on_figure": displayed_on_figure,
        "figure_kind": figure_kind,
        "perlin_modified": perlin_modified,
    })



def build_pos_calibration_from_saved_uniform(path_load, N_particles, M_total, R0, v0):
    """Build the post-processing calibration from the saved uniform raw median."""
    file_uniform_sta = path_load + "DiffuStatistics_pos_type_uniform_N{}.txt".format(N_particles)
    data_uniform_sta = np.loadtxt(file_uniform_sta, dtype=float)
    if data_uniform_sta.ndim != 1 or data_uniform_sta.size < 4:
        raise ValueError("unexpected position statistics format: {}".format(file_uniform_sta))
    uniform_median_raw = data_uniform_sta[3]

    file_uniform_pers = path_load + "DiffuPers_pos_type_uniform_N{}.txt".format(N_particles)
    data_uniform_pers = np.loadtxt(file_uniform_pers, dtype=float)
    q50 = data_uniform_pers[np.isclose(data_uniform_pers[:, 0], 0.5), 1]
    if q50.size != 1 or not np.isclose(q50[0], uniform_median_raw, rtol=5.0e-6):
        raise ValueError("position statistics median and saved q=0.5 disagree for N={}".format(N_particles))

    return dbc.build_classical_limit_calibration(
        N_particles, M_total, R0, v0, uniform_median_raw, pos_or_vel="pos"
    )


def prepare_pos_calibration_diagnostics(
    path_load, path_save, N_particles_list, M_total, R0, v0
):
    """Write and validate position calibration diagnostics before any plotting."""
    calibration_by_N = {}
    rows = []
    for N_particles in N_particles_list:
        calibration_info = build_pos_calibration_from_saved_uniform(
            path_load, N_particles, M_total, R0, v0
        )
        calibration_by_N[int(N_particles)] = calibration_info

        modified_stats_file = path_load + "DiffuStatistics_pos_type_noised_N{}.txt".format(
            N_particles
        )
        modified_stats = np.loadtxt(modified_stats_file, dtype=float)
        if modified_stats.ndim != 1 or modified_stats.size < 4:
            raise ValueError("unexpected position statistics format: {}".format(modified_stats_file))
        modified_median_raw = float(modified_stats[3])
        uniform_median_raw = calibration_info["reference_median_raw"]
        uniform_median_calibration = float(
            dbc.apply_diffusion_calibration(uniform_median_raw, calibration_info)
        )
        modified_median_calibration = float(
            dbc.apply_diffusion_calibration(modified_median_raw, calibration_info)
        )
        zeta_raw = modified_median_raw / uniform_median_raw
        zeta_calibration = modified_median_calibration / uniform_median_calibration
        reference_relative_error = abs(
            uniform_median_calibration / calibration_info["D_classical"] - 1.0
        )
        zeta_absolute_error = abs(zeta_calibration - zeta_raw)

        rows.append([
            N_particles,
            calibration_info["D_classical"],
            uniform_median_raw,
            calibration_info["formula_correction"],
            calibration_info["C_calibration"],
            calibration_info["total_factor"],
            uniform_median_calibration,
            zeta_raw,
            zeta_calibration,
            reference_relative_error,
            zeta_absolute_error,
            float(calibration_info["legacy_position_gxx_correction_enabled"]),
        ])

    rows = np.asarray(rows, dtype=float)
    diagnostic_path = path_save + "calibration_pos.txt"
    np.savetxt(
        diagnostic_path,
        rows,
        header=(
            "N D_classical median_pos_uniform_raw GXX_correction_factor "
            "C_r_after_GXX total_position_calibration_factor "
            "median_pos_uniform_calibration zeta_r_raw zeta_r_calibration "
            "reference_relative_error zeta_absolute_error "
            "legacy_position_gxx_correction_enabled"
        ),
        fmt="%.17e",
    )
    validation = dbc.validate_calibration_diagnostics(
        rows[:, 0], rows[:, 1], rows[:, 6], rows[:, 7], rows[:, 8], rows[:, 5]
    )
    print("position calibration diagnostics passed:", validation)
    print("wrote", diagnostic_path)
    return calibration_by_N, rows, validation


def load_and_plot_pos_workflow(
    path_load, snapshot_name, tag, N_particles, N_iter, is_by_generating,
    calibration_info, is_enable_plot_each=True, fractal_dimension_records=None
):
    tag_N = "_N"+str(N_particles)
    file_vel = None
    if not is_by_generating:
        file_vel = path_load+"xv_normalized_"+snapshot_name+".txt" #by reading
    else:
        file_vel = path_load+"pos_noised"+str(N_iter)+tag_N+".txt" #by generating
    file_nr = path_load+"nr_pos_"+snapshot_name+tag_N+".txt"
    file_diffu_each = path_load+"Diffu_pos_"+snapshot_name+tag_N+".txt"
    file_diffu_sta = path_load+"DiffuStatistics_pos_"+snapshot_name+tag_N+".txt"
    file_diffu_pers = path_load+"DiffuPers_pos_"+snapshot_name+tag_N+".txt"
    suffix = "pos_{}_N{}".format(snapshot_name, N_particles)
    ads.DEBUG_PRINT_V(1, suffix, "suffix")

    vel = None
    data_diffu_eff_each = None
    h_frac, Dim_frac = None, None

    if is_enable_plot_each and N_particles<=100000: #much time to load
        vel = np.loadtxt(file_vel, dtype=float)[:, 0:3]
        # vel_list[i_tag] = vel
        data_diffu_each = np.loadtxt(file_diffu_each, dtype=float)
        data_diffu_eff_each = data_diffu_each[:, 2] #tree method
    
    data_nr = np.loadtxt(file_nr, dtype=float)
    radii, inside_counts = data_nr[:, 0], data_nr[:, 1]
    perlin_modified = tag in PERLIN_MODIFIED_POSITION_TAGS
    plot_fractal_fit = is_enable_plot_each and perlin_modified
    h_frac, Dim_frac = gfs.calculate_mean_neareast_count_load(
        radii, inside_counts, save_path=path_save, suffix=suffix,
        is_plot=plot_fractal_fit
    )
    if plot_fractal_fit:
        append_fractal_dimension_record(
            fractal_dimension_records,
            "data/examples_pos/fractal_dimension_fit_{}.pdf".format(suffix),
            snapshot_name, N_particles, Dim_frac, True,
            "fractal_dimension_fit", perlin_modified,
        )

    if data_diffu_eff_each is not None:
        data_diffu_eff_each = dbc.apply_diffusion_calibration(
            data_diffu_eff_each, calibration_info
        )

    data_diffu_sta = np.loadtxt(file_diffu_sta, dtype=float)
    diffu_eff_mean = float(dbc.apply_diffusion_calibration(data_diffu_sta[2], calibration_info))
    diffu_eff_median = float(dbc.apply_diffusion_calibration(data_diffu_sta[3], calibration_info))
    diffu_eff_0 = calibration_info["D_classical"]
    data_diffu_pers = np.loadtxt(file_diffu_pers, dtype=float)
    data_diffu_pers[:, 1] = dbc.apply_diffusion_calibration(
        data_diffu_pers[:, 1], calibration_info
    )
    # label_list[i_tag] = suffix
    # diffueff_median_list[i_tag] = diffu_eff_median
    # diffueff_mean_list[i_tag] = diffu_eff_mean
    # Dim_frac_list[i_tag] = Dim_frac
    ads.DEBUG_PRINT_V(1, diffu_eff_median, Dim_frac, "Dim_frac")

    if is_enable_plot_each and N_particles<=100000: #much time to plot
        Dim_frac_plot = Dim_frac if perlin_modified else None
        dbc.plot_sample_3D_pos_or_vel(
            vel, path_save, Dim_frac=Dim_frac_plot, suffix=suffix, pos_or_vel="pos"
        )
        dbc.plot_velocity_DF_contour_compare([vel], [suffix], save_path=path_save, pos_or_vel="pos")
        dbc.plot_normalized_pdf_from_eachpoints_histogram_vel_data(
            data_diffu_eff_each, diffu_eff_0, suffix=suffix, pos_or_vel="pos"
        )
        append_fractal_dimension_record(
            fractal_dimension_records,
            "data/examples_pos/samplepoints_pos_{}.pdf".format(suffix),
            snapshot_name, N_particles, Dim_frac, perlin_modified,
            "3D_scatter", perlin_modified,
        )
        append_fractal_dimension_record(
            fractal_dimension_records,
            "data/examples_pos/diffu_r_DF_{}_histogram.pdf".format(suffix),
            snapshot_name, N_particles, Dim_frac, False,
            "diffusion_histogram", perlin_modified,
        )
        # dbc.plot_normalized_pdf_from_eachpoints_KDE(data_diffu_eff_each, diffu_eff_0, suffix=suffix, pos_or_vel="pos")
        # dbc.plot_normalized_pdf_from_percentile(
        #     data_diffu_pers, diffu_eff_mean, diffu_eff_median, diffu_eff_0, suffix=suffix, pos_or_vel="pos"
        # )
    return h_frac, Dim_frac, diffu_eff_mean, diffu_eff_median



## main
if __name__ == '__main__':

    # Process only saved raw diffusion data unless the optional historical
    # observed/simulated branches are explicitly enabled below.
    ## total positions
    #### Step 1. settings
    R0 = dbc.R0_scale
    Rm = R0/dbc.lambda2
    M_total = dbc.M_total_gal_1e10MSun
    v0 = np.sqrt(dbc.G*M_total/dbc.frac_mass/R0) #set the typical speed as the virial speed
    v2m_sqrt = v0 #set the mean speed as the virial speed
    
    vmean_3 = np.array([ #the setted tree mean velocity of velocity DF
        0., 0., 0.
    ])
    v2m = v2m_sqrt*v2m_sqrt
    dispersion_3_iso_ratio = np.array([1., 1., 1.]) #the setted iso diag-component rate
    dispersion_3_high_ratio = np.array([8., 6., 1.]) #the setted aniso diag-component rate

    sig_iso = dbc.calculate_dispersion_iso(v2m, vmean_3)
    dispersion_3_iso = ads.rescale_dispersion_keep_ratio(vmean_3, dispersion_3_iso_ratio, v2m_sqrt)
    dispersion_3_high = ads.rescale_dispersion_keep_ratio(vmean_3, dispersion_3_high_ratio, v2m_sqrt)
    beta_z = 1.-(dispersion_3_high[0]**2+dispersion_3_high[1]**2)/(2.*dispersion_3_high[2]**2)
    ads.DEBUG_PRINT_V(1, M_total, R0, v0, v2m_sqrt, "galaxy_info basic")
    ads.DEBUG_PRINT_V(1, dispersion_3_iso, dispersion_3_high, beta_z, "galaxy_info dispersion")

    #### Step 2. samples and diffusion coefficients
    #: settings
    path_load = "../data/samples_pos/"
    path_save = "../data/examples_pos/"
    # is_run_exe = True
    is_run_exe = False
    is_enable_plot_each = True
    # is_enable_plot_each = False
    is_enable_plot_eta = True
    # is_enable_plot_eta = False
    # is_run_different_types = True
    is_run_different_types = False

    # N_particles_list = [10000]
    # N_particles_list = [10000, 20000, 40000]
    # N_particles_list = [40000]
    N_particles_list = [10000, 20000, 40000, 100000, 1000000, 10000000] #at PC
    # N_particles_list = [10000, 100000, 1000000, 10000000, 100000000, 1000000000] #at server
    # np.random.seed(75)
    pos_DFs = [
        "pos_uniform", "pos_uniform_noise", "pos_obs", "pos_simu", 
        # "pos_extra", 
        # "pos_crystal", "pos_obs", 
        # "pos_simu", "pos_simu_noise", 
        # "pos_obs", "pos_uniform", "pos_uniform_noise", "pos_simu", "pos_simu_noise", 
    ]

    pos_uniform = None
    pos_uniform_noise = None
    pos_obs = None
    pos_simu = None
    pos_extra = None
    pos_info_store_file = "../data/pos_info_store.csv"
    pos_info_store = []
    tag_type_list = ["uniform", "noised", "simulated", "obs"]

    diffueff_uniform_median_list = None
    diffueff_noised_median_list = None
    diffueff_uniform_mean_list = None
    diffueff_noised_mean_list = None
    diffueff_observed_list = None
    diffueff_simulated_list = None
    diffueff_referencevalue_list = None

    h_frac_list = np.zeros((len(N_particles_list), len(tag_type_list)))
    Dim_frac_list = np.zeros_like(h_frac_list)
    diffu_eff_mean_list = np.zeros_like(h_frac_list)
    diffu_eff_median_list = np.zeros_like(h_frac_list)
    fractal_dimension_records = []
    calibration_by_N, calibration_rows_pos, calibration_validation_pos = (
        prepare_pos_calibration_diagnostics(
            path_load, path_save, N_particles_list, M_total, R0, v0
        )
    )

    pos_simu_original = None
    pos_simu_original_ubstable = None
    pos_obs_original = None
    if is_run_different_types:
        snapshot_ID = 120
        data_path = "../data/samples_simulated/snapshot_%03d_example_merge.txt" % snapshot_ID
        data = np.loadtxt(data_path, dtype=float)
        pos_simu_original = data[:, 0:3].copy()
        ads.DEBUG_PRINT_V(
            1, np.shape(pos_simu_original), dci.get_mean_radius(pos_simu_original),
            "pos_simu_original"
        )

        snapshot_ID = 40
        data_path = "../data/samples_simulated/snapshot_%03d_example_merge.txt" % snapshot_ID
        data = np.loadtxt(data_path, dtype=float)
        pos_simu_original_ubstable = data[:, 0:3].copy()
        data = None
        ads.DEBUG_PRINT_V(
            1, np.shape(pos_simu_original_ubstable),
            dci.get_mean_radius(pos_simu_original_ubstable),
            "pos_simu_original_ubstable"
        )

        pos_obs_original = pd.read_csv(odp.file_path_6D_Cartesian).to_numpy()[:, 0:3]
        ads.DEBUG_PRINT_V(
            1, np.shape(pos_obs_original), dci.get_mean_radius(pos_obs_original),
            "pos_obs_original"
        )

    for (i_np, N_particles) in enumerate(N_particles_list):

        # #: get data and plot each sample
        # b90 = R0 / (lambda4 * N_particles)
        # coef_const_Diffu = dbc.calculate_coef_const_Diffu(M_total, N_particles, R0, lambda4)
        # v_epsilon = dbc.calculate_v_epsilon(v2m_sqrt, lambda4, N_particles)
        # coef_const_Diffu_vel = dbc.calculate_coef_const_Diffu_vel(M_total, N_particles, R0, lambda4)
        # ads.DEBUG_PRINT_V(1, N_particles, v_epsilon, coef_const_Diffu_vel, "galaxy_info N_particles")

        #: call C++ prog to generate samples and calculate diffusion coef
        #: note that one should make before calling C++
        
        #: basic types
        #: (1) pos_uniform (noised0) by generating
        i_tag = 0
        tag = tag_type_list[i_tag]

        exe_path = "../diffu_r_simple_each/./out.exe"
        snapshot_name = "type_{}".format(tag)
        N_particles = N_particles #user set
        M_total_set = M_total #user set
        R0_set = R0 #user set
        # is_by_generating = 0
        is_by_generating = 1
        is_diffu = 1
        N_iter = 0 #user set
        # N_iter = 14 #user set
        vmean_3_x = vmean_3[0] #user set
        vmean_3_y = vmean_3[1] #user set
        vmean_3_z = vmean_3[2] #user set
        ratio_sigma_xx = dispersion_3_high_ratio[0] #user set
        ratio_sigma_yy = dispersion_3_high_ratio[1] #user set
        ratio_sigma_zz = dispersion_3_high_ratio[2] #user set

        file_vel_input = path_load+"xv_"+snapshot_name+".txt"

        if is_run_exe:
            dcc.run_cpp_program_pos(
                exe_path, snapshot_name, N_particles, M_total_set, R0_set, is_by_generating, is_diffu, N_iter, 
                vmean_3_x, vmean_3_y, vmean_3_z, ratio_sigma_xx, ratio_sigma_yy, ratio_sigma_zz, 
                path_save+"nohup.out"
            )

        calibration_info_pos = calibration_by_N[int(N_particles)]
        print("position calibration:", calibration_info_pos)
        
        h_frac_list[i_np, i_tag], Dim_frac_list[i_np, i_tag], diffu_eff_mean_list[i_np, i_tag], diffu_eff_median_list[i_np, i_tag] = \
            load_and_plot_pos_workflow(
                path_load, snapshot_name, tag, N_particles, N_iter, is_by_generating,
                calibration_info_pos, is_enable_plot_each=is_enable_plot_each,
                fractal_dimension_records=fractal_dimension_records
            )

        #: (2) a. pos_noised (noised14) by generating
        i_tag = 1
        tag = tag_type_list[i_tag]

        exe_path = "../diffu_r_simple_each/./out.exe"
        snapshot_name = "type_{}".format(tag)
        N_particles = N_particles #user set
        M_total_set = M_total #user set
        R0_set = R0 #user set
        # is_by_generating = 0
        is_by_generating = 1
        is_diffu = 1
        # N_iter = 0 #user set
        N_iter = 14 #user set
        vmean_3_x = vmean_3[0] #user set
        vmean_3_y = vmean_3[1] #user set
        vmean_3_z = vmean_3[2] #user set
        ratio_sigma_xx = dispersion_3_high_ratio[0] #user set
        ratio_sigma_yy = dispersion_3_high_ratio[1] #user set
        ratio_sigma_zz = dispersion_3_high_ratio[2] #user set

        file_vel_input = path_load+"xv_"+snapshot_name+".txt"

        if is_run_exe:
            dcc.run_cpp_program_pos(
                exe_path, snapshot_name, N_particles, M_total_set, R0_set, is_by_generating, is_diffu, N_iter, 
                vmean_3_x, vmean_3_y, vmean_3_z, ratio_sigma_xx, ratio_sigma_yy, ratio_sigma_zz, 
                path_save+"nohup.out"
            )
        
        h_frac_list[i_np, i_tag], Dim_frac_list[i_np, i_tag], diffu_eff_mean_list[i_np, i_tag], diffu_eff_median_list[i_np, i_tag] = \
            load_and_plot_pos_workflow(
                path_load, snapshot_name, tag, N_particles, N_iter, is_by_generating,
                calibration_info_pos, is_enable_plot_each=is_enable_plot_each,
                fractal_dimension_records=fractal_dimension_records
            )
        
        #: (2) b. pos_noised (noised14) by generating, illustration
        if N_particles == 40000:
            i_tag = 1
            tag = tag_type_list[i_tag]

            exe_path = "../diffu_r_simple_each/./out.exe"
            snapshot_name = "type_{}_largenoise".format(tag)
            N_particles = N_particles #user set
            M_total_set = M_total #user set
            R0_set = R0 #user set
            # is_by_generating = 0
            is_by_generating = 1
            is_diffu = 1
            # N_iter = 0 #user set
            N_iter = 50 #user set
            vmean_3_x = vmean_3[0] #user set
            vmean_3_y = vmean_3[1] #user set
            vmean_3_z = vmean_3[2] #user set
            ratio_sigma_xx = dispersion_3_high_ratio[0] #user set
            ratio_sigma_yy = dispersion_3_high_ratio[1] #user set
            ratio_sigma_zz = dispersion_3_high_ratio[2] #user set

            file_vel_input = path_load+"xv_"+snapshot_name+".txt"

            if is_run_exe:
                dcc.run_cpp_program_pos(
                    exe_path, snapshot_name, N_particles, M_total_set, R0_set, is_by_generating, is_diffu, N_iter, 
                    vmean_3_x, vmean_3_y, vmean_3_z, ratio_sigma_xx, ratio_sigma_yy, ratio_sigma_zz, 
                    path_save+"nohup.out"
                )
            
            h_frac_tmp, Dim_frac_tmp, diffu_mean_tmp, diffu_median_tmp = \
                load_and_plot_pos_workflow(
                    path_load, snapshot_name, tag, N_particles, N_iter, is_by_generating,
                    calibration_info_pos, is_enable_plot_each=is_enable_plot_each,
                    fractal_dimension_records=fractal_dimension_records
                )
        
        #: other types
        if is_run_different_types:
            #: (3) a. pos_simulated by random selecting
            i_tag = 2
            tag = tag_type_list[i_tag]

            exe_path = "../diffu_r_simple_each/./out.exe"
            snapshot_name = "type_{}".format(tag)
            N_particles = N_particles #user set
            M_total_set = M_total #user set
            R0_set = R0 #user set
            is_by_generating = 0
            # is_by_generating = 1
            is_diffu = 1
            N_iter = 0 #user set
            # N_iter = 14 #user set
            vmean_3_x = vmean_3[0] #user set
            vmean_3_y = vmean_3[1] #user set
            vmean_3_z = vmean_3[2] #user set
            ratio_sigma_xx = dispersion_3_high_ratio[0] #user set
            ratio_sigma_yy = dispersion_3_high_ratio[1] #user set
            ratio_sigma_zz = dispersion_3_high_ratio[2] #user set

            file_vel_input = path_load+"xv_"+snapshot_name+".txt"
            pos_simu = ads.random_choice_without_prior(pos_simu_original, N_particles)
            v_sample = np.zeros((N_particles, 3))
            xv_sample = np.hstack((pos_simu, v_sample))
            np.savetxt(file_vel_input, xv_sample)

            if is_run_exe:
                dcc.run_cpp_program_pos(
                    exe_path, snapshot_name, N_particles, M_total_set, R0_set, is_by_generating, is_diffu, N_iter, 
                    vmean_3_x, vmean_3_y, vmean_3_z, ratio_sigma_xx, ratio_sigma_yy, ratio_sigma_zz, 
                    path_save+"nohup.out"
                )
            
            h_frac_list[i_np, i_tag], Dim_frac_list[i_np, i_tag], diffu_eff_mean_list[i_np, i_tag], diffu_eff_median_list[i_np, i_tag] = \
                load_and_plot_pos_workflow(
                    path_load, snapshot_name, tag, N_particles, N_iter, is_by_generating,
                    calibration_info_pos, is_enable_plot_each=is_enable_plot_each,
                    fractal_dimension_records=fractal_dimension_records
                )
            
            #: (3) b. pos_simulated by random selecting, illustration
            if N_particles == 40000:
                i_tag = 2
                tag = tag_type_list[i_tag]

                exe_path = "../diffu_r_simple_each/./out.exe"
                snapshot_name = "type_{}_unstable".format(tag)
                N_particles = N_particles #user set
                M_total_set = M_total #user set
                R0_set = R0 #user set
                is_by_generating = 0
                # is_by_generating = 1
                is_diffu = 1
                N_iter = 0 #user set
                # N_iter = 14 #user set
                vmean_3_x = vmean_3[0] #user set
                vmean_3_y = vmean_3[1] #user set
                vmean_3_z = vmean_3[2] #user set
                ratio_sigma_xx = dispersion_3_high_ratio[0] #user set
                ratio_sigma_yy = dispersion_3_high_ratio[1] #user set
                ratio_sigma_zz = dispersion_3_high_ratio[2] #user set

                file_vel_input = path_load+"xv_"+snapshot_name+".txt"
                pos_simu = ads.random_choice_without_prior(pos_simu_original_ubstable, N_particles)
                v_sample = np.zeros((N_particles, 3))
                xv_sample = np.hstack((pos_simu, v_sample))
                np.savetxt(file_vel_input, xv_sample)

                if is_run_exe:
                    dcc.run_cpp_program_pos(
                        exe_path, snapshot_name, N_particles, M_total_set, R0_set, is_by_generating, is_diffu, N_iter, 
                        vmean_3_x, vmean_3_y, vmean_3_z, ratio_sigma_xx, ratio_sigma_yy, ratio_sigma_zz, 
                        path_save+"nohup.out"
                    )
                
                h_frac_tmp, Dim_frac_tmp, diffu_mean_tmp, diffu_median_tmp = \
                    load_and_plot_pos_workflow(
                        path_load, snapshot_name, tag, N_particles, N_iter, is_by_generating,
                        calibration_info_pos, is_enable_plot_each=is_enable_plot_each,
                        fractal_dimension_records=fractal_dimension_records
                    )
            
            #: (4) pos_obs by random selecting
            i_tag = 3
            tag = tag_type_list[i_tag]

            exe_path = "../diffu_r_simple_each/./out.exe"
            snapshot_name = "type_{}".format(tag)
            N_particles = N_particles #user set
            M_total_set = M_total #user set
            R0_set = R0 #user set
            is_by_generating = 0
            # is_by_generating = 1
            is_diffu = 1
            N_iter = 0 #user set
            # N_iter = 14 #user set
            vmean_3_x = vmean_3[0] #user set
            vmean_3_y = vmean_3[1] #user set
            vmean_3_z = vmean_3[2] #user set
            ratio_sigma_xx = dispersion_3_high_ratio[0] #user set
            ratio_sigma_yy = dispersion_3_high_ratio[1] #user set
            ratio_sigma_zz = dispersion_3_high_ratio[2] #user set

            file_vel_input = path_load+"xv_"+snapshot_name+".txt"
            pos_obs = ads.random_choice_without_prior(pos_obs_original, N_particles)
            v_sample = np.zeros((N_particles, 3))
            xv_sample = np.hstack((pos_obs, v_sample))
            np.savetxt(file_vel_input, xv_sample)

            if is_run_exe:
                dcc.run_cpp_program_pos(
                    exe_path, snapshot_name, N_particles, M_total_set, R0_set, is_by_generating, is_diffu, N_iter, 
                    vmean_3_x, vmean_3_y, vmean_3_z, ratio_sigma_xx, ratio_sigma_yy, ratio_sigma_zz, 
                    path_save+"nohup.out"
                )
            
            h_frac_list[i_np, i_tag], Dim_frac_list[i_np, i_tag], diffu_eff_mean_list[i_np, i_tag], diffu_eff_median_list[i_np, i_tag] = \
                load_and_plot_pos_workflow(
                    path_load, snapshot_name, tag, N_particles, N_iter, is_by_generating,
                    calibration_info_pos, is_enable_plot_each=is_enable_plot_each,
                    fractal_dimension_records=fractal_dimension_records
                )
        
        print("N_particles {}, end.".format(N_particles))
        # exit(0) #debug

    # Replot TeX-referenced historical samples directly from their saved raw
    # files.  This deliberately bypasses the old random-resampling branch so
    # data/samples_pos remains unchanged.
    saved_paper_samples = [
        ("type_noised_largenoise", "noised", 10000, 50, True),
        ("type_obs", "obs", 40000, 0, False),
        ("type_simulated", "simulated", 40000, 0, False),
        ("type_simulated_unstable", "simulated", 40000, 0, False),
    ]
    for snapshot_name, tag, N_particles, N_iter, is_by_generating in saved_paper_samples:
        print("Replot saved paper sample", snapshot_name, "N", N_particles)
        load_and_plot_pos_workflow(
            path_load, snapshot_name, tag, N_particles, N_iter, is_by_generating,
            calibration_by_N[int(N_particles)],
            is_enable_plot_each=is_enable_plot_each,
            fractal_dimension_records=fractal_dimension_records,
        )

    dbc.update_fractal_dimension_figure_index(
        fractal_dimension_records, FRACTAL_DIMENSION_INDEX_PATH, "position"
    )

    #: plot eta_N
    diffueff_median_p_uniform = copy.deepcopy(diffu_eff_median_list[:, 0])
    diffueff_median_p_noised = copy.deepcopy(diffu_eff_median_list[:, 1])
    diffueff_mean_p_uniform = copy.deepcopy(diffu_eff_mean_list[:, 0])
    diffueff_mean_p_noised = copy.deepcopy(diffu_eff_mean_list[:, 1])

    h_frac_mean_uniform = copy.deepcopy(np.mean(h_frac_list[:, 0]))
    h_frac_mean_noised = copy.deepcopy(np.mean(h_frac_list[:, 1]))
    Dim_frac_mean_uniform = copy.deepcopy(np.mean(Dim_frac_list[:, 0]))
    Dim_frac_mean_noised = copy.deepcopy(np.mean(Dim_frac_list[:, 1]))
    
    ads.DEBUG_PRINT_V(1, diffu_eff_mean_list[:, 0], diffu_eff_mean_list[:, 1], "diffu_eff_mean_list")
    ads.DEBUG_PRINT_V(1, diffu_eff_median_list[:, 0], diffu_eff_median_list[:, 1], "diffu_eff_median_list")
    ads.DEBUG_PRINT_V(1, Dim_frac_list[:, 0], Dim_frac_list[:, 1], "Dim_frac_list")
    
    if is_enable_plot_eta:
        N_particles_list = np.array(N_particles_list)
        hu = h_frac_mean_uniform
        du = Dim_frac_mean_uniform
        hn = h_frac_mean_noised
        dn = Dim_frac_mean_noised
        h0 = None
        d0 = 3.
        Dim_frac_lb = [dn, du, d0]
        # ads.DEBUG_PRINT_V(0, Dim_frac_lb)
        
        diffueff_uniform_median_list = [ #note: a pointer for array, need to copy
            N_particles_list, diffueff_median_p_uniform, hu, du, "pos_uniform", 
        ]
        diffueff_noised_median_list = [
            N_particles_list, diffueff_median_p_noised, hu, dn, "pos_noised", 
        ]

        diffueff_uniform_mean_list = [ #note: a pointer for array, need to copy
            N_particles_list, diffueff_mean_p_uniform, hu, du, "pos_uniform", 
        ]
        diffueff_noised_mean_list = [
            N_particles_list, diffueff_mean_p_noised, hu, dn, "pos_noised", 
        ]

        suffix = "modified_pos"
        dbc.plot_relaxation_time_with_N_and_dim_pos(
            R0, v0, M_total, Dim_frac_lb, 
            diffueff_uniform_median_list, diffueff_uniform_mean_list, 
            diffueff_noised_median_list, diffueff_noised_mean_list, 
            diffueff_observed_list, diffueff_simulated_list, 
            diffueff_referencevalue_list, suffix
        )

    #: calculate amplification ratio zeta for pos
    zeta_pos_type = np.zeros_like(diffu_eff_median_list)
    for i in range(len(tag_type_list)):
        zeta_pos_type[:, i] = diffu_eff_median_list[:, i]/diffu_eff_median_list[:, 0]
    zeta_versus_N = np.hstack((np.array([N_particles_list]).T, zeta_pos_type))
    np.savetxt(path_save+"zeta_pos_type.txt", zeta_versus_N)

    zeta_pos = zeta_pos_type[:, 1]
    zeta_versus_N = np.hstack((np.array([N_particles_list]).T, np.array([zeta_pos]).T))
    np.savetxt(path_save+"zeta_pos.txt", zeta_versus_N)
    ads.DEBUG_PRINT_V(1, zeta_versus_N, "zeta_pos")

    dbc.zeta_linear_fitting_outer(
        N_particles_list, zeta_pos, [1.0e11],
        save_path=path_save, suffix="zeta_pos"
    )
