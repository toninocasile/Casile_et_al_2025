'''
This file plots the results of the shuffling analysis produced, by setting the appropriate
variables, by the file process_IFPs_responsive_AND_selective_cond_1_vs_cond_2_multiple_contrasts.py

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
from collections import namedtuple
# This is to test for bimodality
import diptest

from matplotlib import colors
import matplotlib.patches as mpatches

# This is for violinplot
import seaborn as sns

import matplotlib
if os.name == 'nt':
    # This works best under Windows
    matplotlib.use('Qt5Agg')
else:
    # This is for Linux. If ones keeps this setting also for Windows then there is memory leakage when plotting
    # figures in a loop and used memory keeps increasing at each cycle
    matplotlib.use('TKAgg')

# set interactive mode for plotting
plt.ion()

# import location of directories
from utilities import results_dir, figures_dir

# save_figs == True --> figures are also saved on disk
save_figs = False

# number of significant electrodes across all contrasts
n_significant_by_contrasts = {'mirc_vs_submirc':156, 'submirc_vs_submirc_post':39}
# These data are used to set the axis limits
lim_x_axis_by_contrast = {'mirc_vs_submirc':170, 'submirc_vs_submirc_post':60}

if False:
    # set this branch to True to plot results of the shuffling analysis for the MIRC vs subMIRC condition
    res_file_name = 'results_responsive_AND_selective__mirc_vs_submirc_shuffle_NON_parametric_tests_remove_outliers_single_electrode__ 50-550ms.pickle'
else:
    # set this branch to True to plot results of the shuffling analysis for the subMIRC post vs subMIRC condition
    res_file_name = 'results_responsive_AND_selective__submirc_vs_submirc_post_shuffle_NON_parametric_tests_remove_outliers_single_electrode__ 50-550ms.pickle'

# load results of the shuffling analysis
in_file = open(os.path.join(results_dir, res_file_name), 'rb')
[conds_names, cond_1_vs_cond_2_subjs_shufflings, curr_shuffle_ind, \
 cond_1_trial_inds_subjs, cond_2_trial_inds_subjs, \
 all_cond_1_trial_inds_subjs, all_cond_2_trial_inds_subjs, n_consecutive_pvals] = pickle.load(in_file)
in_file.close()
n_contrasts = cond_1_vs_cond_2_subjs_shufflings.shape[0]

# set contrast names
contrasts_names = []
for curr_contrast_ind in range(0, n_contrasts):
    contrasts_names.append(conds_names[curr_contrast_ind][0] + '_vs_' + conds_names[curr_contrast_ind][1])

# total electrodes
total_eletrodes_shufflings = np.sum(cond_1_vs_cond_2_subjs_shufflings[ : ,0:curr_shuffle_ind-1], axis=0)

# now plot each distribution separately
for curr_contrast_ind, curr_contrast_name in enumerate(contrasts_names):
    # extract the data
    curr_shufflings = cond_1_vs_cond_2_subjs_shufflings[curr_contrast_ind, 0:curr_shuffle_ind]
    # get 95% interval
    curr_shuffled_data_modulus_95_quantile = np.quantile(curr_shufflings, 0.95)

    # ... and plot the data
    fig, axes = plt.subplots()
    n, bins, rectangles = axes.hist(curr_shufflings, bins=25, density=True, label='bootstrapping distribution', color='dimgray')
    x_lims = [0, lim_x_axis_by_contrast[curr_contrast_name]]
    y_lims = axes.get_ylim()
    delta_x = x_lims[1] - curr_shuffled_data_modulus_95_quantile
    delta_y = y_lims[1] - y_lims[0]
    axes.add_patch(
        matplotlib.patches.Rectangle((curr_shuffled_data_modulus_95_quantile, 0.0), delta_x, delta_y, facecolor='red',
                                     alpha=0.2, label='95% quantile'))
    # now plot line with median distance of submirc post vs submirc to mirc vs submirc electrodes
    axes.plot([n_significant_by_contrasts[curr_contrast_name], n_significant_by_contrasts[curr_contrast_name]], y_lims, linewidth=3, color='black', linestyle='--', \
              label='total # of electrodes (data={0:d})'.format(n_significant_by_contrasts[curr_contrast_name]))
    # beautify the plot
    axes.tick_params(axis='both', labelsize=16)
    # axes.legend(fontsize=12, loc='center right')
    # axes.set_xlim(x_lims[0], x_lims[1])
    axes.set_xlim(x_lims)
    axes.set_ylim(y_lims)
    axes.set_xlabel('# of significant electrodes', fontsize=18)
    axes.set_ylabel('probability density function', fontsize=18)
    separator = '__'
    contrasts_string = separator.join(contrasts_names)
    axes.set_title('{0} contrasts - shuffle analysis'.format(curr_contrast_name), fontsize=8)
    plt.tight_layout()

    if save_figs == True:
        f_name = os.path.join(figures_dir, 'bootstrap_analysis_HISTOGRAM_{0}'.format(curr_contrast_name))
        # now save figure
        fig.savefig(fname=f_name + '.png', dpi=800)
        # fig.savefig(fname=f_name + '.eps', dpi=800)
        fig.savefig(fname=f_name + '.svg', dpi=800)


print('DONE!')