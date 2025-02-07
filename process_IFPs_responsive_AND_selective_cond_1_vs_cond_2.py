'''
Here I process the IFPs for MIRC, sub-MIRC, sub-MIRC post and object stimuli

This file is the "omnibus" processing file that represents the first step in our pipeline analysis

Data are processed for the paper:

Neural correlates of minimal recognizable configurations in the human brain
by Casile et al.

author:
Antonino Casile
University of Messina
antonino.casile@unime.it
toninocasile@gmail.com
'''

# Some standard imports
import os
import sys
import time
import warnings
import pandas as pd
import scipy.io
import scipy.stats as stats
import random
import pickle
import scipy
import scipy.fft
import scipy.signal as signal
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import namedtuple
from parse import parse
import datetime
from enum import Enum

# This is to load Matlab files
from scipy.io import loadmat

# This is for listing directories' content using wildcards
import glob

import matplotlib

# This is for multiprocessing
import multiprocessing

# import stuff from utilities
from utilities import res_analysis, results_dir, behavior_dir, ifps_base_dir, electrodes_location_dir, get_bad_channels

# initialize random number generator (it is used for the shuffle analysis)
random.seed(time.time())

if True:
    # ------------------------ Settings for MIRC vs SUBMIRC --------------------------------------
    conds_codes = ((2, 1), )
    conds_names = (('mirc', 'submirc'), )
    conds_codes_for_shuffling = [1, 2]
else:
    # ------------------------ Settings for SUBMIRC vs SUBMIRC_POST ------------------------------
    conds_codes = ((1, (5, 8, 11)),)
    conds_names = (('submirc', 'submirc_post'), )
    conds_codes_for_shuffling = [1, 5, 8, 11]

# This is a sanity check
n_contrasts = len(conds_codes)
for curr_contrast_ind in range(0, n_contrasts):
    if len(conds_codes[curr_contrast_ind]) != 2:
        raise Exception('ERROR: Contrast #{0} -- There are {1} conditions to analyze. I am expecting TWO'.\
                        format(curr_contrast_ind, len(conds_codes)))

# subject files
subj_files = ['subj1_behavior','subj2_behavior', 'subj3_behavior', 'subj4_behavior', 'subj5_behavior', 'subj6_behavior', 'subj7_behavior',
              'subj8_behavior', 'subj9_behavior', 'subj10_behavior', 'subj11_behavior', 'subj12_behavior']
subj_sample_Hz = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
n_subjs = len(subj_files)

# temporal interval of the extracted trial intervals
trial_int_s = [-.2, 1]

# p-value used as threshold and number of consecutive bins in which t-test < pval_thresh
pval_thresh = 0.01
n_consecutive_pval = 50

IFPs_band_Hz = [0.1, 100.0]

# intervals (IN SECONDS) used to compare conditions
conds_compare_interval_s = [0.05, 0.55]

# baseline interval
baseline_interval_s = [-0.2, 0.05]

'''
shuffle_trials = True --> shuffle trials.

Set this variable to True to perform an analysis in which categories labels are shuffled and
then the number of responsive and selective electrodes is computed. This process is repeated
n_shufflings times to compute the "null-hypothesis" distribution of the number of responsive
and selective electrodes
'''
shuffle_trials = True
if shuffle_trials == True:
    n_shufflings = 500
else:
    n_shufflings = 1

# redirect_console == True --> console output is redirected to save the log on file
redirect_console = False
if shuffle_trials == True:
    # When we are shuffling trials we DO NOT redirect the console messages to file
    redirect_console = False

# defines whether we use parallel toolbox to process data
# Set this variable to True if you have a multi-core CPU, setting the nCores accordingly
useParallel = True

# number of cores used for processing data
nCores = 15

# ------------------------------------------------------------------------------------------
# ------------------------------------------ CODE ------------------------------------------
# ------------------------------------------------------------------------------------------
"""
This function takes as input a vector of 0's and 1's and returns ALL the indices
of the intervals containing more than n_ones consecutive ones
inds_interval = interval of indices in which consecutive ones MUST start to be considere valid
"""
def find_consecutive_ones(p_vals, pval_thresh, n_consecutive, inds_interval, times_ms):
    # differentiate to find start and ends of significant intervals
    tmp = np.diff(np.concatenate(([0], p_vals < pval_thresh, [0])))

    # initialize output variables
    significant = False
    el_begin_ms = []
    el_end_ms = []

    # find where Boolean values change and the length of the associated interval
    inds_up = np.where(tmp == 1)[0]
    inds_down = np.where(tmp == -1)[0]
    # Since we add a value at the end of tmp then if the difference is on that "final" value we have to
    # bring the index to the size of current_pvals
    inds_down[inds_down >= len(p_vals)] =  len(p_vals) - 1
    diff_inds = inds_down - inds_up

    # check if some interval is longer than our threshold
    interval_inds = np.where(diff_inds >= n_consecutive)[0]

    if len(interval_inds) > 0:
        # check that at least one interval is in the correct range
        for curr_interval_ind in range(0, len(interval_inds)):
            if (inds_up[interval_inds[curr_interval_ind]] >= inds_interval[0]) & (
                    inds_up[interval_inds[curr_interval_ind]] < inds_interval[1]):
                el_begin_ms.append(times_ms[inds_up[interval_inds[curr_interval_ind]]])
                el_end_ms.append(times_ms[inds_down[interval_inds[curr_interval_ind]]])
                significant = True
                # break

    return significant, el_begin_ms, el_end_ms

"""
This function finds and returns the anatomical locations of the electrode given as input.
This function returns the following tuple):
    (electrodes_pos, electrodes_region)
    
    elec_name is in the form, e.g. 'subj6_ch_023'
    
"""
def find_electrode_info(elec_name, electrodes_location_dir):
    # parse string to extract subject's and electrode's number
    parse_res = parse('subj{}_ch_{}', elec_name)
    curr_subj_num = int(parse_res[0])
    # get electrode's number
    curr_el_number = int(parse_res[1])

    # read file with electrodes' locations
    f_name_localization = os.path.join(electrodes_location_dir, 'subj{0}_localization.xlsx'.format(curr_subj_num))
    electrodes_df = pd.read_excel(f_name_localization)
    electrodes_number = list(electrodes_df['channel_nr'])
    all_electrodes_area = list(electrodes_df['area'])
    all_electrodes_hemisphere = list(electrodes_df['hemisphere'])

    # read electrodes positions. For some files the column is 'MNI coordinates avg' for others 'coordinates avg'
    if 'MNI coordinates avg' in electrodes_df.columns:
        all_electrodes_pos = electrodes_df['MNI coordinates avg']
    # For subj5 the average brain coordinates are in the field 'to avg brain coordinates'
    elif 'to avg brain coordinates' in electrodes_df.columns:
        all_electrodes_pos = electrodes_df['to avg brain coordinates']
    else:
        raise Warning('Could not find electrodes!')

    if curr_el_number in electrodes_number:
        # get electrode's index
        el_index = electrodes_number.index(curr_el_number)
        # get and parse the string representing the position of the current electrode
        tmp_pos = parse('{} {} {}', all_electrodes_pos[el_index])
        electrode_pos = np.array([float(tmp_pos[0]), float(tmp_pos[1]), float(tmp_pos[2])])
        # There are electrodes for which, in the localization file, there is the position BUT NOT the brain region
        # In these cases the 'macro area' field is set to 'nan' which is a float
        # These lines are needed to take this condition into account
        if isinstance(electrodes_df['macro area'][el_index], str):
            electrode_region = electrodes_df['macro area'][el_index]
        else:
            electrode_region = 'NA_POS'
        electrode_area = all_electrodes_area[el_index]
        electrode_hemisphere = int(all_electrodes_hemisphere[el_index])
    else:
        electrode_pos = np.array([float('nan'), float('nan'), float('nan')])
        electrode_region = 'NA'
        electrode_area = 'NA'
        electrode_hemisphere = int(-1)

    # return electrode's position and region
    return electrode_pos, electrode_region, electrode_area, electrode_hemisphere

# ------------------------------- Here starts the code -------------------------------

"""
This function computes the number of significant channels for one subject

PARS:
    IFPs_band_Hz = [inf_Hz, sup_Hz] --> parameters used for filtering. If either value is None then the corresponding
        filtering (inf_Hz=high pass and sup_Hz = low pass) is not performed
    conds_compare_interval_s --> interval, from trial start, in which condition 1 (e.g. mirc) and
    condition 2 (e.g. submirc) responses are compared.
    baseline_compare_interval_s --> interval, from trial start, used to compute baseline (e.g. [-0.2, 0])
"""
def compute_n_significant(args_in):
    # unpack input arguments
    (curr_subj_str, in_curr_subj_behavioral_codes, shuffle_trials, f_name_behavior, f_name_ifps, subj_sample_Hz, IFPs_band_Hz,
     conds_codes, conds_names, conds_codes_for_shuffling, conds_compare_interval_s, baseline_compare_interval_s) = args_in

    print('--------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------')
    # get subject number
    tmp = parse('subj{}', curr_subj_str)
    curr_subj_num = int(tmp[0])

    print('Computing subject {0}'.format(curr_subj_str))

    # load IFPs for the current subject
    tmp_ifps = loadmat(f_name_ifps)
    IFPs = tmp_ifps['Total_Epoch_Data']
    tmp_channel_names = tmp_ifps['electrodes_names'][0]
    # reformat list of electrodes' names into a list of strings
    channel_names = [el[0] for el in tmp_channel_names]
    # delete variables that we no longer need
    del tmp_ifps

    # perform a sanity check
    if len(channel_names) != IFPs.shape[0]:
        raise Warning('len(channel_names_in) != IFPs.shape[0]')

    # ----------------- filter IFPs --------------------------
    if IFPs_band_Hz[1] != -1:
        b, a = scipy.signal.butter(5, IFPs_band_Hz[1] / (subj_sample_Hz / 2), btype='low')
        IFPs = scipy.signal.filtfilt(b, a, IFPs, axis=-1)
    IFPs_original = IFPs.copy()

    # get indices of bad channels
    bad_channels_num = get_bad_channels(curr_subj_num)
    if len(bad_channels_num) > 0:
        # now find indices of bad channels
        bad_channels_ind = len(bad_channels_num)*[None]
        for curr_bad_channel_ind, curr_bad_channel_num in enumerate(bad_channels_num):
            for curr_channel_name_ind, curr_channel_name in enumerate(channel_names):
                if curr_channel_name.find('ch_{0:03d}'.format(curr_bad_channel_num)) > 0:
                    break
            bad_channels_ind[curr_bad_channel_ind] = curr_channel_name_ind
    else:
        bad_channels_ind = []

    # We print diagnostic messages only if we are NOT performing a shuffling analysis
    if shuffle_trials == False:
        # print excluded electrodes
        print('------ Excluded electrodes (bad channels) -----------')
        if len(bad_channels_ind) > 0:
            for curr_ind in bad_channels_ind:
                print(channel_names[curr_ind])
        else:
            print('NO EXCLUDED CHANNEL')

    # Here we compute the indices that we have to retain
    sel_electrodes_inds = list(set(range(0, len(channel_names))).difference(set(bad_channels_ind)))

    # perform a sanity check
    if IFPs.shape[0] != len(channel_names):
        raise Exception('file {0}: number of channels DOES NOT MATCH the length of the list of channel names'.format(f_name_ifps))

    # Initialize output variables
    # ----------------------- Variables for SIGNIFICANT AND SELECTIVE electrodes -----------------------
    responsive_selective_n_chans = 0
    responsive_selective_chans_name = []
    responsive_selective_chans_pos = np.zeros((IFPs.shape[0], 3))
    responsive_selective_chans_hemisphere = np.zeros(IFPs.shape[0])
    responsive_selective_chans_area = []
    responsive_selective_chans_macro_area = []
    responsive_selective_time_begin_ms = []
    responsive_selective_time_end_ms = []
    # ----------------------- Variables for RESPONSIVE ONLY electrodes -----------------------
    responsive_n_chans = 0
    responsive_chans_name = []
    responsive_chans_pos = np.zeros((IFPs.shape[0], 3))
    responsive_chans_hemisphere = np.zeros(IFPs.shape[0])
    responsive_chans_area = []
    responsive_chans_macro_area = []
    responsive_time_begin_ms = []
    responsive_time_end_ms = []
    # ----------------------- Variables for SELECTIVE ONLY electrodes -----------------------
    selective_n_chans = 0
    selective_chans_name = []
    selective_chans_pos = np.zeros((IFPs.shape[0], 3))
    selective_chans_hemisphere = np.zeros(IFPs.shape[0])
    selective_chans_area = []
    selective_chans_macro_area = []
    selective_time_begin_ms = []
    selective_time_end_ms = []

    # get number of electrodes
    subj_electrodes_num = IFPs.shape[0]

    # extract trials for condition 1 and 2 respectively. In some case, we transform everything into tuples
    # because one condition can have more than one code associated with it (e.g. for submirc-post)
    if type(conds_codes[0]) is not tuple:
        curr_cond = (conds_codes[0], )
    cond_1_trial_inds_orig = [inds for (inds, val) in enumerate(in_curr_subj_behavioral_codes) if val in curr_cond]
    if type(conds_codes[1]) is not tuple:
        curr_cond = (conds_codes[1],)
    else:
        curr_cond = conds_codes[1]
    cond_2_trial_inds_orig = [inds for (inds, val) in enumerate(in_curr_subj_behavioral_codes) if val in curr_cond]

    # times (in ms) of each trial
    times_s = np.linspace(start=trial_int_s[0], stop=trial_int_s[1], num=IFPs.shape[-1])
    times_ms = 1000 * times_s

    # compute indices of the minimum and maximum values of the interval used to compare conditions
    tmp = np.abs(times_s - conds_compare_interval_s[0])
    inds1 = np.where(tmp == np.min(tmp))[0][0]
    tmp = np.abs(times_s - conds_compare_interval_s[1])
    inds2 = np.where(tmp == np.min(tmp))[0][0]
    conds_compare_interval_ind = [inds1, inds2]

    # compute indices of the minimum and maximum values of the interval used to compare conditions
    tmp = np.abs(times_s - baseline_compare_interval_s[0])
    inds1 = np.where(tmp == np.min(tmp))[0][0]
    tmp = np.abs(times_s - baseline_compare_interval_s[1])
    inds2 = np.where(tmp == np.min(tmp))[0][0]
    baseline_interval_ind = [inds1, inds2]

    # ----------------------- Subtract baseline -----------------------
    for curr_electrode_ind in range(0, subj_electrodes_num):
        # indices of trials for the present condition
        cond_1_trial_inds = cond_1_trial_inds_orig.copy()
        cond_2_trial_inds = cond_2_trial_inds_orig.copy()

        # We print stuff on the console only if we are not performing a shuffling analysis
        if (curr_electrode_ind % 10) == 0:
            print('Subject {0} - Processing electrode {1}'.format(curr_subj_str, curr_electrode_ind))

        # extract trials for current electrode
        IFPs_trials = np.squeeze(IFPs_original[curr_electrode_ind, :, :])

        # compute mean power across ALL trials
        IFPs_power = np.mean(IFPs_trials ** 2, axis=1)
        mean_power_across_all_trials = np.mean(IFPs_power)
        std_power_across_all_trials = np.std(IFPs_power)

        # find trials whose power for the current electrode are 4 std away from the mean
        excluded_trials = \
        np.where(np.transpose(IFPs_power) > (mean_power_across_all_trials + 4 * std_power_across_all_trials))[0]

        if len(excluded_trials) > 0:
            cond_1_trial_inds = list(set(cond_1_trial_inds).difference(set(excluded_trials)))
            cond_1_trial_inds.sort()
            cond_2_trial_inds = list(set(cond_2_trial_inds).difference(set(excluded_trials)))
            cond_2_trial_inds.sort()

        # compare number of trials from behavioral and trial data
        n_trials_behavior = len(in_curr_subj_behavioral_codes)
        n_trials_IFPs = len(IFPs_trials)
        # ... and perform a sanity check
        if (n_trials_behavior != n_trials_IFPs):
            raise Warning('subjet {0} -- n_trials_behavior != n_trials_IFPs'.format(subj_str))

        # extract cond_1 and cond_2 trials
        in_cond_1_trials = IFPs_trials[cond_1_trial_inds, :]
        in_cond_2_trials = IFPs_trials[cond_2_trial_inds, :]

        # extract baseline intervals for both MIRC and SUBMIRC trials
        baseline_cond_1 = in_cond_1_trials[:, baseline_interval_ind[0]:baseline_interval_ind[1]]
        baseline_cond_2 = in_cond_2_trials[:, baseline_interval_ind[0]:baseline_interval_ind[1]]

        # remove average across the baseline from all trials
        cond_1_trials = in_cond_1_trials - np.mean(baseline_cond_1)
        cond_2_trials = in_cond_2_trials - np.mean(baseline_cond_2)

        # save data "back" into the IFPs matrix
        IFPs[curr_electrode_ind, cond_1_trial_inds, :] = cond_1_trials
        IFPs[curr_electrode_ind, cond_2_trial_inds, :] = cond_2_trials

    # now shuffle trials
    curr_subj_behavioral_codes = in_curr_subj_behavioral_codes.copy()
    if shuffle_trials == True:
        inds = [curr_ind for curr_ind in range(0, len(curr_subj_behavioral_codes)) if
                curr_subj_behavioral_codes[curr_ind] in conds_codes_for_shuffling]
        inds_shuffled = inds.copy()
        random.shuffle(inds_shuffled)
        curr_subj_behavioral_codes[inds] = curr_subj_behavioral_codes[inds_shuffled]

        # extract trials for condition 1 and 2 respectively. In some case, we transform everything into tuples
        # because one condition can have more than one code associated with it (e.g. for submirc-post)
        if type(conds_codes[0]) is not tuple:
            curr_cond = (conds_codes[0], )
        s_cond_1_trial_inds_orig = [inds for (inds, val) in enumerate(curr_subj_behavioral_codes) if val in curr_cond]
        if type(conds_codes[1]) is not tuple:
            curr_cond = (conds_codes[1],)
        else:
            curr_cond = conds_codes[1]
        s_cond_2_trial_inds_orig = [inds for (inds, val) in enumerate(curr_subj_behavioral_codes) if val in curr_cond]
    else:
        s_cond_1_trial_inds_orig = cond_1_trial_inds_orig.copy()
        s_cond_2_trial_inds_orig = cond_2_trial_inds_orig.copy()

    # This is a sanity check. If we shuffle the trials, the sum of their indices must not change
    if len(s_cond_1_trial_inds_orig) + len(s_cond_2_trial_inds_orig) != \
            len(cond_1_trial_inds_orig) + len(cond_2_trial_inds_orig):
        raise Warning('SHUFFLING: the number of shuffled trials IS NOT EQUAL to the number of NON shuffled trials')

    # ----------------------- Here we go processing each electrode -----------------------
    for curr_electrode_ind in range(0, subj_electrodes_num):
        # reset trials to ALL trials
        cond_1_trial_inds = s_cond_1_trial_inds_orig.copy()
        cond_2_trial_inds = s_cond_2_trial_inds_orig.copy()

        if shuffle_trials == False:
            # We print stuff on the console only if we are not performing a shuffling analysis
            if curr_electrode_ind % 20 == 0:
                print('Processing electrode {0}'.format(curr_electrode_ind))

        # find trials whose power exceeds 4std from the mean

        # extract trials for current electrode
        IFPs_trials = np.squeeze(IFPs_original[curr_electrode_ind, :, :])

        # compute mean power across ALL trials
        IFPs_power = np.mean(IFPs_trials ** 2, axis=1)
        mean_power_across_all_trials = np.mean(IFPs_power)
        std_power_across_all_trials = np.std(IFPs_power)

        # find trials whose power FOR THE CURRENT ELECTRODE are 4 std away from the mean
        excluded_trials = np.where(np.transpose(IFPs_power) > (mean_power_across_all_trials + 4 * std_power_across_all_trials))[0]

        if len(excluded_trials) > 0:
            cond_1_trial_inds = list(set(cond_1_trial_inds).difference(set(excluded_trials)))
            cond_1_trial_inds.sort()
            cond_2_trial_inds = list(set(cond_2_trial_inds).difference(set(excluded_trials)))
            cond_2_trial_inds.sort()

        # compare number of trials from behavioral and trial data
        n_trials_behavior = len(curr_subj_behavioral_codes)
        n_trials_IFPs = len(IFPs_trials)
        # ... and perform a sanity check
        if (n_trials_behavior != n_trials_IFPs):
            raise Warning('subjet {0} -- n_trials_behavior != n_trials_IFPs'.format(subj_str))

        # extract cond_1 and cond_2 trials
        cond_1_trials = IFPs[curr_electrode_ind, cond_1_trial_inds, :]
        cond_2_trials = IFPs[curr_electrode_ind, cond_2_trial_inds, :]

        # We initialize some variables that we need further down
        significant = False
        el_begin_ms = []
        el_end_ms = []

        # We perform analysis only if the current electrode IS NOT marked "bad channel"
        # initialize variables
        if curr_electrode_ind in sel_electrodes_inds:
            # perform statistics
            pvals_cond_1 = stats.wilcoxon(cond_1_trials).pvalue
            pvals_cond_2 = stats.wilcoxon(cond_2_trials).pvalue
            pvals_cond_1_vs_cond_2 = scipy.stats.mannwhitneyu(cond_1_trials, cond_2_trials, axis=0)[1]
            # check if there are n_consecutive_pval consecutive elements in which the test is passed
            responsive_cond_1, el_begin_ms_cond_1, el_end_ms_cond_1 = \
                find_consecutive_ones(pvals_cond_1, pval_thresh, n_consecutive_pval, conds_compare_interval_ind, times_ms)
            responsive_cond_2, el_begin_ms_cond_2, el_end_ms_cond_2 = \
                find_consecutive_ones(pvals_cond_2, pval_thresh, n_consecutive_pval, conds_compare_interval_ind, times_ms)
            responsive_cond_1_vs_cond_2, el_begin_ms_cond_1_vs_cond_2, el_end_ms_cond_cond_1_vs_cond_2 = \
                find_consecutive_ones(pvals_cond_1_vs_cond_2, pval_thresh, n_consecutive_pval, conds_compare_interval_ind,
                                      times_ms)

            # ----------------------------- RESPONSIVE ONLY channels -----------------------------
            # if significant append current electrode to the list of significant electrodes
            if ((responsive_cond_1 == True) or (responsive_cond_2 == True)):
                responsive_n_chans = responsive_n_chans + 1
                responsive_chans_name.append(channel_names[curr_electrode_ind])
                responsive_time_begin_ms.append([min(el_begin_ms_cond_1 + el_begin_ms_cond_2)])
                responsive_time_end_ms.append([max(el_end_ms_cond_1 + el_end_ms_cond_2)])

                # get information for the current significant electrode and store them
                curr_free_ind = len(responsive_chans_name) - 1
                electrode_pos, electrode_macro_area, electrode_area, electrode_hemisphere = \
                    find_electrode_info(channel_names[curr_electrode_ind], electrodes_location_dir)
                responsive_chans_hemisphere[curr_free_ind] = electrode_hemisphere
                responsive_chans_pos[curr_free_ind, :] = electrode_pos
                responsive_chans_area.append(electrode_area)
                responsive_chans_macro_area.append(electrode_macro_area)

            # ----------------------------- SELECTIVE ONLY channels -----------------------------
            # if significant append current electrode to the list of significant electrodes
            if responsive_cond_1_vs_cond_2:
                selective_n_chans = selective_n_chans + 1
                selective_chans_name.append(channel_names[curr_electrode_ind])
                selective_time_begin_ms.append(el_begin_ms_cond_1_vs_cond_2)
                selective_time_end_ms.append(el_end_ms_cond_cond_1_vs_cond_2)

                # get information for the current significant electrode and store them
                curr_free_ind = len(selective_chans_name) - 1
                electrode_pos, electrode_macro_area, electrode_area, electrode_hemisphere = \
                    find_electrode_info(channel_names[curr_electrode_ind], electrodes_location_dir)
                selective_chans_hemisphere[curr_free_ind] = electrode_hemisphere
                selective_chans_pos[curr_free_ind, :] = electrode_pos
                selective_chans_area.append(electrode_area)
                selective_chans_macro_area.append(electrode_macro_area)

            # ----------------------------- RESPONSIVE AND SELECTIVE channels -----------------------------
            # if significant append current electrode to the list of significant electrodes
            if responsive_cond_1_vs_cond_2 and ((responsive_cond_1 == True) or (responsive_cond_2 == True)):
                responsive_selective_n_chans = responsive_selective_n_chans + 1
                responsive_selective_chans_name.append(channel_names[curr_electrode_ind])
                # responsive_selective_time_begin_ms.append(np.min(el_begin_ms_cond_1 + el_begin_ms_cond_2))
                # responsive_selective_time_end_ms.append(np.min(el_end_ms_cond_1 + el_end_ms_cond_2))
                responsive_selective_time_begin_ms.append(el_begin_ms_cond_1_vs_cond_2)
                responsive_selective_time_end_ms.append(el_end_ms_cond_cond_1_vs_cond_2)

                # get information for the current significant electrode and store them
                curr_free_ind = len(responsive_selective_chans_name)-1
                electrode_pos, electrode_macro_area, electrode_area, electrode_hemisphere = \
                    find_electrode_info(channel_names[curr_electrode_ind], electrodes_location_dir)
                responsive_selective_chans_hemisphere[curr_free_ind] = electrode_hemisphere
                responsive_selective_chans_pos[curr_free_ind, :] = electrode_pos
                responsive_selective_chans_area.append(electrode_area)
                responsive_selective_chans_macro_area.append(electrode_macro_area)

    # SELECTIVE AND RESPONSIVE: remove elements that were not used
    if len(responsive_selective_chans_name) > 0:
        responsive_selective_chans_pos = responsive_selective_chans_pos[0:len(responsive_selective_chans_name), :]
        responsive_selective_chans_hemisphere = responsive_selective_chans_hemisphere[0:len(responsive_selective_chans_name)]

    # pack results into a named tuple
    res_responsive_selective = res_analysis(responsive_selective_n_chans, responsive_selective_chans_name, \
                                            responsive_selective_time_begin_ms, responsive_selective_time_end_ms, \
                                            responsive_selective_chans_pos, responsive_selective_chans_area, \
                                             responsive_selective_chans_macro_area, responsive_selective_chans_hemisphere)


    # RESPONSIVE: remove elements that were not used
    if len(responsive_chans_name):
        responsive_chans_pos = responsive_chans_pos[0:len(responsive_chans_name), :]
        responsive_chans_hemisphere = responsive_chans_hemisphere[0:len(responsive_chans_name)]

    # pack results into a named tuple
    res_responsive = res_analysis(responsive_n_chans, responsive_chans_name, \
                                            responsive_time_begin_ms, responsive_time_end_ms, \
                                            responsive_chans_pos, responsive_chans_area, \
                                             responsive_chans_macro_area, responsive_chans_hemisphere)

    # SELECTIVE: remove elements that were not used
    if len(selective_chans_name):
        selective_chans_pos = selective_chans_pos[0:len(selective_chans_name), :]
        selective_chans_hemisphere = selective_chans_hemisphere[0:len(selective_chans_name)]

    # pack results into a named tuple
    res_selective = res_analysis(responsive_n_chans, selective_chans_name, \
                                            selective_time_begin_ms, selective_time_end_ms, \
                                            selective_chans_pos, selective_chans_area, \
                                             selective_chans_macro_area, selective_chans_hemisphere)

    # return number of significant electrode
    return res_responsive_selective, res_responsive, res_selective, cond_1_trial_inds, cond_2_trial_inds, cond_1_trial_inds_orig, cond_2_trial_inds_orig

# When using multiprocessing I have to use this, since each process will run the "parent" file and we do not want
# that each process spawns again all processes
if __name__ == '__main__':
    if redirect_console == True:
        # build file name based on the computations that were performed
        f_name = 'log_responsive_'
        for curr_contrast_ind in range(0, n_contrasts):
            f_name_redirect = f_name + '_{0}_OR_{1}'.format(conds_names[curr_contrast_ind][0], conds_names[curr_contrast_ind][1])

        # log also remove outliers
        f_name_redirect = f_name_redirect + '_NON_parametric_tests_remove_outliers_single_electrode'

        f_name_redirect = f_name_redirect + '__{0:3d}-{1:3d}ms.txt'.\
            format(int(conds_compare_interval_s[0]*1000), int(conds_compare_interval_s[1]*1000))

    # -----------------------------------------------
    n_cond_1_vs_cond_2_subjs = np.zeros([n_contrasts, n_subjs])
    # n_electrodes_subjs = np.zeros([n_contrasts, n_subjs])

    # ------------------- variables for RESPONSIVE AND SELECTIVE CHANNELS -------------------
    res_responsive_subjs = [None]*n_contrasts
    for curr_contrast_ind in range(0, n_contrasts):
        res_responsive_subjs[curr_contrast_ind] = [None] * n_subjs

    # ------------------- variables for SELECTIVE ONLY CHANNELS -------------------
    res_selective_subjs = [None]*n_contrasts
    for curr_contrast_ind in range(0, n_contrasts):
        res_selective_subjs[curr_contrast_ind] = [None] * n_subjs

    # ------------------- variables for RESPONSIVE ONLY CHANNELS -------------------
    res_responsive_selective_subjs = [None]*n_contrasts
    for curr_contrast_ind in range(0, n_contrasts):
        res_responsive_selective_subjs[curr_contrast_ind] = [None] * n_subjs


    # ------------------- variables to save the indices of used trials -------------------
    cond_1_trial_inds_subjs = [None] * n_contrasts
    cond_2_trial_inds_subjs = [None] * n_contrasts
    all_cond_1_trial_inds_subjs = [None] * n_contrasts
    all_cond_2_trial_inds_subjs = [None] * n_contrasts
    for curr_contrast_ind in range(0, n_contrasts):
        cond_1_trial_inds_subjs[curr_contrast_ind] = [None] * n_subjs
        cond_2_trial_inds_subjs[curr_contrast_ind] = [None] * n_subjs
        all_cond_1_trial_inds_subjs[curr_contrast_ind] = [None] * n_subjs
        all_cond_2_trial_inds_subjs[curr_contrast_ind] = [None] * n_subjs


    # here we save the number of responsive and significant electrodes for each shuffling
    cond_1_vs_cond_2_subjs_shufflings = np.zeros([n_contrasts, n_shufflings])
    # here we save the number of significant electrodes FOR EACH subject and for each shuffling
    cond_1_vs_cond_2_subjs_electrodes_shufflings = np.zeros([n_contrasts, n_shufflings, n_subjs])

    # check whether we have to redirect console output to file
    if redirect_console == True:
        default_stdout = sys.stdout
        stdout_file = open(os.path.join(results_dir, f_name_redirect), 'w')
        sys.stdout = stdout_file

    for curr_shuffle_ind in range(0, n_shufflings):
        if shuffle_trials == True:
            print('-------------- Current Shuffling {0}'.format(curr_shuffle_ind))

        # initialize lists in case we have to use multiprocessing
        all_pars = [[] for _ in range(0, n_contrasts)]

        # ---------------------------- current subject ----------------------------
        for curr_subj_ind in range(0, n_subjs):
        # for curr_subj_ind in range(0, 2):
            print('Processing subject {0}'.format(curr_subj_ind + 1))

            # split string to get currect subject string
            curr_subj_str, tt = subj_files[curr_subj_ind].split('_')

            # file name for behavioral data
            f_name_behavior = os.path.join(behavior_dir, subj_files[curr_subj_ind]) + '_Tonino.xlsx'

            # file name for ifps data
            f_name_ifps = os.path.join(ifps_base_dir, 'subj{0}_all_channels.mat'.format(curr_subj_ind + 1))

            # load behavioral data for the current subject
            behavior_data = pd.read_excel(f_name_behavior)
            curr_subj_behavioral_codes = behavior_data.RelabeledCond_Tonino.copy()
            original_curr_subj_behavioral_codes = behavior_data.Cond.copy()
            curr_subj_behavioral_codes = behavior_data.RelabeledCond_Tonino.copy()

            # call function to compute the number of significant electrodes for the current subject
            if useParallel == True:
                for curr_contrast_ind in range(0, n_contrasts):
                    all_pars[curr_contrast_ind].append((curr_subj_str, curr_subj_behavioral_codes, shuffle_trials, f_name_behavior, f_name_ifps, \
                             subj_sample_Hz[curr_subj_ind], IFPs_band_Hz, conds_codes[curr_contrast_ind], conds_names[curr_contrast_ind], \
                                               conds_codes_for_shuffling, conds_compare_interval_s, baseline_interval_s))
            else:
                for curr_contrast_ind in range(0, n_contrasts):
                    res_responsive_selective, res_responsive, res_selective, \
                        cond_1_trial_inds, cond_2_trial_inds, cond_1_trial_inds_orig, cond_2_trial_inds_orig = \
                        compute_n_significant((curr_subj_str, curr_subj_behavioral_codes, shuffle_trials, f_name_behavior, f_name_ifps, \
                        subj_sample_Hz[curr_subj_ind], IFPs_band_Hz, conds_codes[curr_contrast_ind], conds_names[curr_contrast_ind], \
                                               conds_codes_for_shuffling, conds_compare_interval_s, baseline_interval_s))
                    # accumulate number of significant electrodes per subject
                    n_cond_1_vs_cond_2_subjs[curr_contrast_ind][curr_subj_ind] = len(res_responsive_selective.chans_name)

            if useParallel == False:
                # accumulate values across subjects
                res_responsive_selective_subjs[curr_contrast_ind][curr_subj_ind] = res_responsive_selective
                res_responsive_subjs[curr_contrast_ind][curr_subj_ind] = res_responsive
                res_selective_subjs[curr_contrast_ind][curr_subj_ind] = res_selective
                cond_1_trial_inds_subjs[curr_contrast_ind][curr_subj_ind] = cond_1_trial_inds
                cond_2_trial_inds_subjs[curr_contrast_ind][curr_subj_ind] = cond_2_trial_inds
                all_cond_1_trial_inds_subjs[curr_contrast_ind][curr_subj_ind] = cond_1_trial_inds_orig
                all_cond_2_trial_inds_subjs[curr_contrast_ind][curr_subj_ind] = cond_2_trial_inds_orig

        # Let's start the parallel processes in case we are using the parallel toolbox
        # In this case we do not accumulate results across subjects as this is done automatically by multiprocessing.pool
        if useParallel == True:
            for curr_contrast_ind in range(0, n_contrasts):
                pool = multiprocessing.Pool(processes=nCores)
                tmp_res_responsive_selective, tmp_res_responsive, tmp_res_selective,\
                    tmp_cond_1_trial_inds, tmp_cond_2_trial_inds, tmp_cond_1_trial_inds_orig, \
                    tmp_cond_2_trial_inds_orig = zip(*pool.map(compute_n_significant, all_pars[curr_contrast_ind]))

                # wait for all processes in the Pool to finish and that their used resources are freed
                pool.close()
                pool.join()

                # accumulate values across contrasts!
                res_responsive_selective_subjs[curr_contrast_ind] = tmp_res_responsive_selective
                res_responsive_subjs[curr_contrast_ind] = tmp_res_responsive
                res_selective_subjs[curr_contrast_ind] = tmp_res_selective

                n_chans_subj = [curr_val.n_chans for curr_val in tmp_res_responsive_selective]
                cond_1_trial_inds_subjs[curr_contrast_ind] = tmp_cond_1_trial_inds
                cond_2_trial_inds_subjs[curr_contrast_ind] = tmp_cond_2_trial_inds
                all_cond_1_trial_inds_subjs[curr_contrast_ind] = tmp_cond_1_trial_inds_orig
                all_cond_2_trial_inds_subjs[curr_contrast_ind] = tmp_cond_2_trial_inds_orig

                # In the case of shuffling ONLY the cond_1_vs_cond_2_subjs_shufflings variable makes sense!
                # accumulate results across shufflings
                cond_1_vs_cond_2_subjs_shufflings[curr_contrast_ind, curr_shuffle_ind] = np.sum(n_chans_subj)
                print('Contrasts = {0} -- shufflings=\n{1}'.format(conds_names, cond_1_vs_cond_2_subjs_shufflings))
                n_cond_1_vs_cond_2_subjs[curr_contrast_ind, :] = n_chans_subj


        # ---------------------- SIGNIFICANT AND SELECTIVE: Here we save the results ------------------------
        if shuffle_trials == True:
            f_name = 'results_responsive_AND_selective_'
            for curr_contrast_ind in range(0, n_contrasts):
                f_name = f_name + '_{0}_vs_{1}'.format(conds_names[curr_contrast_ind][0], conds_names[curr_contrast_ind][1])
            f_name = f_name + '_shuffle_NON_parametric_tests_remove_outliers_single_electrode'

            f_name = f_name + '__{0:3d}-{1:3d}ms'. \
                format(int(conds_compare_interval_s[0] * 1000), int(conds_compare_interval_s[1] * 1000))

            # save results in pickle format
            out_file = open(os.path.join(results_dir, f_name + '.pickle'), 'wb')
            pickle.dump([conds_names, cond_1_vs_cond_2_subjs_shufflings, curr_shuffle_ind, \
                         cond_1_trial_inds_subjs, cond_2_trial_inds_subjs, \
                         all_cond_1_trial_inds_subjs, all_cond_2_trial_inds_subjs, n_consecutive_pval], out_file)
            out_file.close()

            print('DONE!')

        else:
            # build file name based on the computations that were performed
            # If that was not a shuffle analysis then we save all the files individually
            for curr_contrast_ind in range(0, n_contrasts):
                f_name = 'results_responsive_AND_selective_'
                f_name = f_name + '_{0}_vs_{1}'.format(conds_names[curr_contrast_ind][0], conds_names[curr_contrast_ind][1])
                if shuffle_trials == True:
                    f_name = f_name + '_shuffle'
                else:
                    f_name = f_name + '_NO_shuffle'
                f_name = f_name + '_NON_parametric_tests_remove_outliers_single_electrode'
                f_name = f_name + '__{0:3d}-{1:3d}ms'.\
                    format(int(conds_compare_interval_s[0]*1000), int(conds_compare_interval_s[1]*1000))

                # save results in pickle format
                out_file = open(os.path.join(results_dir, f_name + '.pickle'), 'wb')
                pickle.dump([res_responsive_selective_subjs, res_responsive_subjs, res_selective_subjs, \
                             cond_1_vs_cond_2_subjs_shufflings[curr_contrast_ind, :], \
                             np.squeeze(cond_1_vs_cond_2_subjs_electrodes_shufflings[curr_contrast_ind, :, :]), \
                             curr_shuffle_ind, n_shufflings, \
                             cond_1_trial_inds_subjs[curr_contrast_ind], cond_2_trial_inds_subjs[curr_contrast_ind], \
                             all_cond_1_trial_inds_subjs[curr_contrast_ind], all_cond_2_trial_inds_subjs[curr_contrast_ind]], \
                            out_file)
                out_file.close()

                # if we are NOT performing a shuffle analysis then we save the data ALSO in ASCII format so that they
                # can be imported in Matlab for plotting
                # ------------------------ results for RESPONSIVE AND SELECTIVE channels ------------------------
                out_file = open(os.path.join(results_dir, f_name + '.txt'), 'w')
                original_stdout = sys.stdout
                sys.stdout = out_file

                for curr_subj_ind in range(0, len(res_responsive_selective_subjs[curr_contrast_ind])):
                    curr_res = res_responsive_selective_subjs[curr_contrast_ind][curr_subj_ind]
                    n_electrodes = len(curr_res.chans_name)
                    if n_electrodes > 0:
                        for curr_electrode_ind in range(0, n_electrodes):
                            print('{0}\t{1:4.2f}\t{2:4.2f}\t{3}\t{4}\t{5}\t{6}'.format(\
                                curr_res.chans_name[curr_electrode_ind], \
                                curr_res.time_begin_ms[curr_electrode_ind][0],\
                                curr_res.time_end_ms[curr_electrode_ind][-1], \
                                  curr_res.chans_macro_area[curr_electrode_ind], \
                                  '{0}\t{1}\t{2}'.format(curr_res.chans_pos[curr_electrode_ind][0], \
                                                         curr_res.chans_pos[curr_electrode_ind][1], \
                                                         curr_res.chans_pos[curr_electrode_ind][2]), \
                                  curr_res.chans_area[curr_electrode_ind], \
                                  int(curr_res.chans_hemisphere[curr_electrode_ind])\
                                ))
                # close file
                out_file.close()

                # ------------------------ results for RESPONSIVE ONLY channels ------------------------
                f_name_responsive = f_name.replace('responsive_AND_selective', 'responsive_ONLY')
                out_file = open(os.path.join(results_dir, f_name_responsive + '.txt'), 'w')
                sys.stdout = out_file

                for curr_subj_ind in range(0, len(res_responsive_subjs[curr_contrast_ind])):
                    curr_res = res_responsive_subjs[curr_contrast_ind][curr_subj_ind]
                    n_electrodes = len(curr_res.chans_name)
                    if n_electrodes > 0:
                        for curr_electrode_ind in range(0, n_electrodes):
                            print('{0}\t{1:4.2f}\t{2:4.2f}\t{3}\t{4}\t{5}\t{6}'.format( \
                                curr_res.chans_name[curr_electrode_ind], \
                                curr_res.time_begin_ms[curr_electrode_ind][0], \
                                curr_res.time_end_ms[curr_electrode_ind][-1], \
                                curr_res.chans_macro_area[curr_electrode_ind], \
                                '{0}\t{1}\t{2}'.format(curr_res.chans_pos[curr_electrode_ind][0], \
                                                       curr_res.chans_pos[curr_electrode_ind][1], \
                                                       curr_res.chans_pos[curr_electrode_ind][2]), \
                                curr_res.chans_area[curr_electrode_ind], \
                                int(curr_res.chans_hemisphere[curr_electrode_ind]) \
                                ))
                # close file
                out_file.close()

                # ------------------------ results for SELECTIVE ONLY channels ------------------------
                f_name_selective = f_name.replace('responsive_AND_selective', 'selective_ONLY')
                out_file = open(os.path.join(results_dir, f_name_selective + '.txt'), 'w')
                sys.stdout = out_file

                for curr_subj_ind in range(0, len(res_selective_subjs[curr_contrast_ind])):
                    curr_res = res_selective_subjs[curr_contrast_ind][curr_subj_ind]
                    n_electrodes = len(curr_res.chans_name)
                    if n_electrodes > 0:
                        for curr_electrode_ind in range(0, n_electrodes):
                            print('{0}\t{1:4.2f}\t{2:4.2f}\t{3}\t{4}\t{5}\t{6}'.format( \
                                curr_res.chans_name[curr_electrode_ind], \
                                curr_res.time_begin_ms[curr_electrode_ind][0], \
                                curr_res.time_end_ms[curr_electrode_ind][-1], \
                                curr_res.chans_macro_area[curr_electrode_ind], \
                                '{0}\t{1}\t{2}'.format(curr_res.chans_pos[curr_electrode_ind][0], \
                                                       curr_res.chans_pos[curr_electrode_ind][1], \
                                                       curr_res.chans_pos[curr_electrode_ind][2]), \
                                curr_res.chans_area[curr_electrode_ind], \
                                int(curr_res.chans_hemisphere[curr_electrode_ind]) \
                                ))
                # close file
                out_file.close()

                # set stdout to its original value
                sys.stdout = original_stdout

    # check whether we have to redirect console output to file
    if redirect_console == True:
        # set sys.stdout back to the default value
        sys.stdout = default_stdout
        stdout_file.close()
        # sys.stdout.close()

    print('DONE!')

