'''
Here I process the IFPs for MIRC, sub-MIRC, sub-MIRC post and object stimuli

This file computes the number of responsive and selective electrodes in the four brain regions (occipital, parietal, temporal and frontal).
It takes as input files processed by process_IFPs_responsive_AND_selective_cond_1_vs_cond_2_multiple_contrasts.py

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
# import mne
import time
import warnings
import pandas as pd
import scipy.io
import scipy.stats as stats
import random
import pickle
# This import is to copy and paste figure
import addcopyfighandler
import scipy
import scipy.fft
import scipy.signal
# import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# This is to test for bimodality
import diptest

from matplotlib import colors

# This is for violinplot
import seaborn as sns

from parse import parse

# This is to load Matlab files
from scipy.io import loadmat

# This is for listing directories' content using wildcards
import glob

import matplotlib
if os.name == 'nt':
    # This works best under Windows
    matplotlib.use('Qt5Agg')
else:
    # This is for Linux. If ones keeps this setting also for Windows then there is memory leakage when plotting
    # figures in a loop and used memory keeps increasing at each cycle
    matplotlib.use('TKAgg')

# activate interactive mode for plotting
plt.ion()

# import stuff from utilities
from utilities import load_results, res_analysis, results_dir, figures_dir, electrodes_location_dir

# redirect_console == True --> console output is redirected to file to save results of data analysis
redirect_console = True

# intervals (IN SECONDS) used to compare conditions
conds_compare_interval_s = [0.05, 0.55]

# mirc vs submirc: threshold used to sort frontal electrodes into "early" and "late" electrodes
threshold_frontal = 420

# save_figs == True --> figures are also saved on disk
save_figs = False

# select the appropriate "if" branch to select a contrast and generate the associated figures and results
if True:
    # settings for mirc vs submirc
    conds_names = ('mirc', 'submirc')
else:
    # settings for mirc vs submirc
    conds_names = ('submirc', 'submirc_post')

# load results
selective_AND_responsive_f_name = 'results_responsive_AND_selective__{0}_vs_{1}_NO_shuffle_NON_parametric_tests_remove_outliers_single_electrode__{2:3d}-{3:3d}ms.txt'.\
    format(conds_names[0], conds_names[1], int(conds_compare_interval_s[0]*1000), int(conds_compare_interval_s[1]*1000))
responsive_ONLY_f_name = 'results_responsive_ONLY__{0}_vs_{1}_NO_shuffle_NON_parametric_tests_remove_outliers_single_electrode__{2:3d}-{3:3d}ms.txt'.\
    format(conds_names[0], conds_names[1], int(conds_compare_interval_s[0]*1000), int(conds_compare_interval_s[1]*1000))

# check whether we have to redirect console output to file
if redirect_console:
    f_name = os.path.join(results_dir, 'STATS_response_times_{0}_vs_{1}_significance_start_end__{2:3d}-{3:3d}ms.txt'.
                          format(conds_names[0], conds_names[1], int(conds_compare_interval_s[0]*1000), int(conds_compare_interval_s[1]*1000)))
    default_stdout = sys.stdout
    stdout_file = open(f_name, 'w')
    sys.stdout = stdout_file

# load SELECTIVE AND RESPONSIVE electrodes
(electrode_names, electrode_time_begin_s, electrode_time_end_s, electrode_macro_area, electrode_pos_t, \
 electrode_area_t, electrode_subj_num, tmp_1) = \
    load_results(os.path.join(results_dir, selective_AND_responsive_f_name), False)
n_significant_electrodes = len(electrode_names)

# load RESPONSIVE ONLY electrodes
(resp_electrode_names, resp_electrode_time_begin_s, resp_electrode_time_end_s, resp_electrode_macro_area, resp_electrode_pos_t, \
 resp_electrode_area, resp_electrode_subj_num, tmp_1) = \
    load_results(os.path.join(results_dir, responsive_ONLY_f_name), False)
n_significant_responsive_electrodes = len(resp_electrode_names)

# compute number of electrodes for each area
macro_area_letters = ('F', 'T', 'P', 'O')
macro_area_names = ('Occipital', 'Frontal', 'Temporal', 'Parietal')
n_responsive_per_macro_area = np.zeros(len(macro_area_names))
n_responsive_per_macro_area = {}
for curr_chan_ind, curr_chan_area_letter in enumerate(macro_area_letters):
    n_responsive_per_macro_area[curr_chan_area_letter] = \
        len(np.where([curr_chan_macro_area == curr_chan_area_letter for curr_chan_macro_area in resp_electrode_macro_area])[0])

# here we start printing our reports
print('Condition {0} vs {1}: total of {2} selective electrodes'.
      format(conds_names[0], conds_names[1], len(electrode_time_begin_s)))

# extract electrodes' significance start and end
time_begin_ms_occipital = np.array([el for ind, el in enumerate(electrode_time_begin_s) if electrode_macro_area[ind]=='O'])
time_begin_ms_frontal = np.array([el for ind, el in enumerate(electrode_time_begin_s) if electrode_macro_area[ind]=='F'])
time_begin_ms_temporal = np.array([el for ind, el in enumerate(electrode_time_begin_s) if electrode_macro_area[ind]=='T'])
time_begin_ms_parietal = np.array([el for ind, el in enumerate(electrode_time_begin_s) if electrode_macro_area[ind]=='P'])

# extract electrodes' subject
occipital_electrodes_subj_num = np.array([subj_num for ind, subj_num in enumerate(electrode_subj_num) if electrode_macro_area[ind]=='O'])
temporal_electrodes_subj_num = np.array([subj_num for ind, subj_num in enumerate(electrode_subj_num) if electrode_macro_area[ind]=='T'])
parietal_electrodes_subj_num = np.array([subj_num for ind, subj_num in enumerate(electrode_subj_num) if electrode_macro_area[ind]=='P'])
frontal_electrodes_subj_num = np.array([subj_num for ind, subj_num in enumerate(electrode_subj_num) if electrode_macro_area[ind]=='F'])

# now perform t-test
print('--------------------- Statistics of the electrodes ---------------------')
n_sel = len(time_begin_ms_occipital)
n_resp = n_responsive_per_macro_area['O']
print('{0} vs {1}: There are {2} selective OCCIPITAL channels out of {3} responsive channels in that area ({4:2.1f}%)'.\
      format(conds_names[0], conds_names[1], n_sel, n_resp, (n_sel / n_resp)*100))
n_sel = len(time_begin_ms_temporal)
n_resp = n_responsive_per_macro_area['T']
print('{0} vs {1}: There are {2} selective TEMPORAL channels out of {3} responsive channels in that area ({4:2.1f}%)'.\
      format(conds_names[0], conds_names[1], n_sel, n_resp, (n_sel / n_resp)*100))
n_sel = len(time_begin_ms_parietal)
n_resp = n_responsive_per_macro_area['P']
print('{0} vs {1}: There are {2} selective PARIETAL channels out of {3} responsive channels in that area ({4:2.1f}%)'.\
      format(conds_names[0], conds_names[1], n_sel, n_resp, (n_sel / n_resp)*100))
n_sel = len(time_begin_ms_frontal)
n_resp = n_responsive_per_macro_area['F']
print('{0} vs {1}: There are {2} selective FRONTAL channels out of {3} responsive channels in that area ({4:2.1f}%)'.\
      format(conds_names[0], conds_names[1], n_sel, n_resp, (n_sel / n_resp)*100))
print('')
print('')
q3, q1 = np.percentile(time_begin_ms_occipital, [75 ,25])
iqr_time_begin_ms_occipital = q3-q1
print('{0} vs {1}: significance_start Median={2}-IQR={3} OCCIPITAL electrodes'.\
      format(conds_names[0], conds_names[1], np.median(time_begin_ms_occipital), iqr_time_begin_ms_occipital))

q3, q1 = np.percentile(time_begin_ms_temporal, [75 ,25])
iqr_time_begin_ms_temporal = q3-q1
print('{0} vs {1}: significance_start median={2}-IQR={3} TEMPORAL electrodes'.\
      format(conds_names[0], conds_names[1], np.median(time_begin_ms_temporal), iqr_time_begin_ms_temporal))

q3, q1 = np.percentile(time_begin_ms_parietal, [75 ,25])
iqr_time_begin_ms_parietal = q3-q1
print('{0} vs {1}: significance_start median={2}-IQR={3} PARIETAL electrodes'.\
      format(conds_names[0], conds_names[1], np.median(time_begin_ms_parietal), iqr_time_begin_ms_parietal))

q3, q1 = np.percentile(time_begin_ms_frontal, [75 ,25])
iqr_time_begin_ms_frontal = q3-q1
print('{0} vs {1}: significance_start median={2}-IQR={3} FRONTAL electrodes'.\
      format(conds_names[0], conds_names[1], np.median(time_begin_ms_frontal), iqr_time_begin_ms_frontal))
print('')

# ------------------ SIGNIFICANCE START -------------------------
# Frontal vs Temporal: perform Mann U
print('Significance start: {0} vs {1}: Comparing Frontal (n={2}) and Temporal (n={3}) responses'.
      format(conds_names[0], conds_names[1], len(time_begin_ms_frontal), len(time_begin_ms_temporal)))
print(stats.mannwhitneyu(time_begin_ms_frontal, time_begin_ms_temporal))
print('')

# Temporal vs Occipital: perform Mann U
print('Significance start: {0} vs {1}: Comparing Occipital (n={2}) and Temporal (n={3}) responses'.
      format(conds_names[0], conds_names[1], len(time_begin_ms_occipital), len(time_begin_ms_temporal)))
print(stats.mannwhitneyu(time_begin_ms_occipital, time_begin_ms_temporal))
print('')

# Occipital vs Frontal: perform Mann U
print('Significance start: {0} vs {1}: Comparing Occipital (n={2}) and Frontal (n={3}) responses'.
      format(conds_names[0], conds_names[1], len(time_begin_ms_occipital), len(time_begin_ms_frontal)))
print(stats.mannwhitneyu(time_begin_ms_occipital, time_begin_ms_frontal))
print('')

# -------------------- DIPTEST (SIGNIFICANCE START) for unimodality of the distribution of the FRONTAL electrodes --------------------
if (conds_names[0] == 'mirc'):
    print('Significance start: {0} vs {1}: diptest - frontal electrodes (n={2})'.format(conds_names[0], conds_names[1], len(time_begin_ms_frontal)))
    print('(val, pvalue)')
    print(diptest.diptest(np.array(time_begin_ms_frontal)))
    print('')
    print('FRONTAL ELECTRODES median={0}'.format(np.median(time_begin_ms_frontal)))

    # Here we check if frontal electrodes are bi-modally distributed and, if so, we perform additional tests
    if (diptest.diptest(np.array(time_begin_ms_frontal))[1] < 0.05):
        print('MIRC vs SUB-MIRC: DIPTEST for FRONTAL electrodes IS SIGNIFICANT')
        print('')
        print('Threshold between early and late frontal responses is {}'.format(threshold_frontal))
        print('')
        tmp_1 = time_begin_ms_frontal[time_begin_ms_frontal < threshold_frontal]
        print('significance start: median of "early" distribution (n={0}) =  {1}'.format(len(tmp_1), np.median(tmp_1)))
        tmp_2 = time_begin_ms_frontal[time_begin_ms_frontal >= threshold_frontal]
        print('significance start: median of "late" distribution (n={0}) =  {1}'.format(len(tmp_2), np.median(tmp_2)))
        print('Significance start: {0} vs {1}: Comparing early frontal (n={2}) and Temporal (n={3}) responses'.
              format(conds_names[0], conds_names[1], len(tmp_1), len(time_begin_ms_temporal)))
        print(stats.mannwhitneyu(tmp_1, time_begin_ms_temporal))
        tmp_3 = time_begin_ms_frontal[time_begin_ms_frontal >= threshold_frontal]
        print('Significance start: {0} vs {1}: Comparing late frontal (n={2}) and Temporal (n={3}) responses'.
              format(conds_names[0], conds_names[1], len(tmp_2), len(time_begin_ms_temporal)))
        print(stats.mannwhitneyu(tmp_3, time_begin_ms_temporal))
        print('')
        print('Significance start: {0} vs {1}: ONE TAILED - Comparing early frontal (n={2}) and Temporal (n={3}) responses'.
              format(conds_names[0], conds_names[1], len(tmp_1), len(time_begin_ms_temporal)))
        print(stats.mannwhitneyu(tmp_1, time_begin_ms_temporal, alternative='less'))

# check whether we have to redirect console output to file
if redirect_console:
    # set sys.stdout back to the default value
    sys.stdout = default_stdout
    stdout_file.close()
    # sys.stdout.close()



# ---------------------------------------------------------------------------
# if we are processing the mirc vs submirc contrast then we also
#  plot histograms of significance_start of FRONTAL ELECTRODES
if conds_names[0] == 'mirc':
    # ----------------- Plot figures for SIGNIFICANCE START: FRONTAL and TEMPORAL -----------------
    y_label = 'start of selectivity (ms)'
    # Select data to plot
    if conds_names[0] == 'mirc':
        # we are plotting mirc vs submirc
        tmp_x = np.concatenate((0 * np.ones(len(time_begin_ms_frontal)), 1 * np.ones(len(time_begin_ms_temporal))))
        tmp_y = np.concatenate((time_begin_ms_frontal, time_begin_ms_temporal))
    else:
        # we are plotting submirc vs submirc_post
        tmp_x = np.concatenate((0 * np.ones(1), 1 * np.ones(len(time_begin_ms_temporal))))
        tmp_y = np.concatenate(([float('NAN')], time_begin_ms_temporal))

    # ... and plot them
    fig, axes = plt.subplots(figsize=(5, 5))
    vplot = sns.violinplot(x=tmp_x, y=tmp_y, ax=axes, color='grey')
    axes.set_xticks([0, 1], labels=['Frontal', 'Temporal'])
    axes.set_ylabel(y_label, fontsize=24)
    axes.tick_params(axis='y', labelsize=22)
    axes.tick_params(axis='x', labelsize=20)
    axes.set_xlim(-0.5, 1.5)
    axes.set_ylim(0, 700)
    plt.tight_layout()

    # now save figure
    if save_figs == True:
        f_name = os.path.join(figures_dir,
                              '{0}_vs_{1}_Frontal_Temporal_violinplot_significance_start'.format(conds_names[0],
                                                                                                 conds_names[1]))
        fig.savefig(fname=f_name + '.png', dpi=300, bbox_inches='tight')
        # fig.savefig(fname=f_name + '.eps', dpi=800, bbox_inches='tight')
        fig.savefig(fname=f_name + '.svg', dpi=300, bbox_inches='tight')

    # extract bins for frontal electrodes
    count_frontal, bins_frontal = np.histogram(time_begin_ms_frontal, range=(50, 550), bins=20)
    time_begin_ms_frontal_early = time_begin_ms_frontal[time_begin_ms_frontal < threshold_frontal]
    time_begin_ms_frontal_late = time_begin_ms_frontal[time_begin_ms_frontal >= threshold_frontal]

    # set colors to plot
    color_frontal = {'early': [1, 0.65, 0], 'late': [0.2, 0.8, 0.2]}

    # ----------------- plot HISTOGRAM of the responses of frontal electrodes: ONE COLOR -----------------
    fig, axes = plt.subplots()
    axes.hist(time_begin_ms_frontal, bins=bins_frontal, color='black')
    axes.set_ylabel('# of electrodes', fontsize=22)
    axes.set_ylim([0, 12])
    axes.set_yticks(range(0, 14, 2))
    axes.set_xlabel('onset of selectivity (ms)', fontsize=22)
    axes.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    # save the figure
    if save_figs == True:
        # now save figure
        f_name = os.path.join(figures_dir, '{0}_vs_{1}_Frontal_HISTOGRAM_ONE_COLOR_significance_start'.format(conds_names[0], conds_names[1]))
        fig.savefig(fname=f_name + '.png', dpi=800, bbox_inches='tight')
        fig.savefig(fname=f_name + '.eps', dpi=800, bbox_inches='tight')
        fig.savefig(fname=f_name + '.svg', dpi=800, bbox_inches='tight')

    # ----------------- plot HISTOGRAM of the responses of frontal electrodes: TWO COLORS -----------------
    fig, axes = plt.subplots()
    axes.hist(time_begin_ms_frontal_early, bins=bins_frontal, color=color_frontal['early'], label='median={0:3.0f}ms'.format(np.median(time_begin_ms_frontal_early)))
    axes.hist(time_begin_ms_frontal_late, bins=bins_frontal, color=color_frontal['late'], label='median={0:3.0f}ms'.format(np.median(time_begin_ms_frontal_late)))
    axes.plot([np.median(time_begin_ms_frontal_early), np.median(time_begin_ms_frontal_early)], [0, 13], linewidth=3, linestyle='--', color='black')
    axes.plot([np.median(time_begin_ms_frontal_late), np.median(time_begin_ms_frontal_late)], [0, 13], linewidth=3, linestyle='--', color='black')
    axes.set_ylabel('# of electrodes', fontsize=22)
    axes.legend(fontsize=15, loc='center left')
    axes.set_yticks(range(0, 14, 2))
    axes.set_ylim([0, 12])
    axes.set_xlabel('start of selectivity (ms)', fontsize=22)
    axes.tick_params(axis='both', labelsize=20)
    plt.tight_layout()

    # save the figure
    if save_figs == True:
        # now save figure
        f_name = os.path.join(figures_dir, '{0}_vs_{1}_Frontal_HISTOGRAM_significance_start_TWO_COLORS'.format(conds_names[0], conds_names[1]))
        fig.savefig(fname=f_name + '.png', dpi=300, bbox_inches='tight')
        # fig.savefig(fname=f_name + '.eps', dpi=800, bbox_inches='tight')
        fig.savefig(fname=f_name + '.svg', dpi=300, bbox_inches='tight')

    # ------------------------ plot sorted ORDERED onset of significance start -------------------------
    # first order times by latency
    sort_inds = np.argsort(time_begin_ms_frontal)
    time_begin_ms_frontal_sorted = time_begin_ms_frontal[sort_inds]
    time_begin_ms_frontal_early = time_begin_ms_frontal_sorted[time_begin_ms_frontal_sorted < threshold_frontal]
    time_begin_ms_frontal_late = time_begin_ms_frontal_sorted[time_begin_ms_frontal_sorted >= threshold_frontal]

    fig, axes = plt.subplots(figsize=(5,5))
    # plot sorted onset of significance
    y_1 = time_begin_ms_frontal_early
    axes.plot(range(0, len(y_1)), time_begin_ms_frontal_early, marker='o', color=color_frontal['early'],
              linestyle='None', label='early')
    axes.plot(range(len(y_1), len(time_begin_ms_frontal)), time_begin_ms_frontal_late, marker='o',
              color=color_frontal['late'], linestyle='None', label='late')

    axes.set_ylabel('onset of significance (ms)', fontsize=20)
    axes.set_xlabel('electrode #', fontsize=20)
    axes.tick_params(axis='both', labelsize=18)
    axes.legend(loc='lower right', fontsize=18)
    plt.tight_layout()

    # set the title of the window's bar
    fig.canvas.manager.set_window_title('{0}_vs_{1}-Frontal electrodes'.format(conds_names[0], conds_names[1]))

    # save the figure
    if save_figs == True:
        # now save figure
        f_name = os.path.join(figures_dir,
                              '{0}_vs_{1}_Frontal_sorted_significance_start_TWO_COLORS'.format(conds_names[0], conds_names[1]))
        fig.savefig(fname=f_name + '.png', dpi=800, bbox_inches='tight')
        fig.savefig(fname=f_name + '.eps', dpi=800, bbox_inches='tight')
        fig.savefig(fname=f_name + '.svg', dpi=800, bbox_inches='tight')

    # ------------------- plot sorted onset of significance start vs subject number -------------------
    # first order times by latency
    sort_inds = np.argsort(time_begin_ms_frontal)

    fig, axes = plt.subplots(1, 2)
    # plot sorted onset of significance
    axes[0].plot(np.array(time_begin_ms_frontal)[sort_inds], marker='o', linestyle='None')
    axes[0].set_ylabel('onset of significance (ms)', fontsize=20)
    axes[0].set_xlabel('electrode #', fontsize=20)
    axes[0].tick_params(axis='both', labelsize=18)
    # plot subject number
    axes[1].plot(np.array(frontal_electrodes_subj_num)[sort_inds], marker='o', linestyle='None')
    axes[1].set_ylabel('subject number', fontsize=20)
    axes[1].set_xlabel('electrode #', fontsize=20)
    axes[1].tick_params(axis='both', labelsize=18)
    plt.tight_layout()

    # save the figure
    if save_figs == True:
        # now save figure
        f_name = os.path.join(figures_dir, '{0}_vs_{1}_Frontal_sorted_significance_start_VS_subjs'.format(conds_names[0], conds_names[1]))
        fig.savefig(fname=f_name + '.png', dpi=800, bbox_inches='tight')
        fig.savefig(fname=f_name + '.eps', dpi=800, bbox_inches='tight')
        fig.savefig(fname=f_name + '.svg', dpi=800, bbox_inches='tight')

# ---------------------------------------------------------------------------
# if we are processing the submirc vs submirc post contrast then we also
#  plot several plots of the temporal electrodes
if conds_names[0] == 'submirc':
    # plot violin plot of ONLY temporal electrodes

    # ----------------- Plot figures for SIGNIFICANCE START: TEMPORAL LOBE -----------------
    y_label = 'start of selectivity (ms)'
    # Select data to plot
    tmp_y = time_begin_ms_temporal
    tmp_x = 0 * np.ones(len(tmp_y))

    # ... and plot them
    fig, axes = plt.subplots(figsize=(5, 5))
    vplot = sns.violinplot(x=tmp_x, y=tmp_y, ax=axes, color='grey')
    axes.set_xticks([0], labels=['Temporal'])
    axes.set_ylabel(y_label, fontsize=24)
    axes.tick_params(axis='y', labelsize=22)
    axes.tick_params(axis='x', labelsize=20)
    axes.set_xlim(-0.5, 0.5)
    axes.set_ylim(0, 700)
    plt.tight_layout()

    # now save figure
    if save_figs == True:
        f_name = os.path.join(figures_dir,
                              '{0}_vs_{1}_Frontal_Temporal_NO_occipital_violinplot_significance_start'.format(
                                  conds_names[0], conds_names[1]))
        fig.savefig(fname=f_name + '.png', dpi=300, bbox_inches='tight')
        # fig.savefig(fname=f_name + '.eps', dpi=800, bbox_inches='tight')
        fig.savefig(fname=f_name + '.svg', dpi=300, bbox_inches='tight')


print('DONE')

