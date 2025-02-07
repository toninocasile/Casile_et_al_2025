'''
This Python processed data for the paper:

Neural correlates of minimal recognizable configurations in the human brain
by Casile et al.

author:
Antonino Casile
University of Messina
antonino.casile@unime.it
toninocasile@gmail.com

'''


# import needed libraries
import os
import sys
import time
import warnings
import pandas as pd

import scipy
import scipy.fft
import scipy.stats as stats
import scipy.signal
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# This package is needed to save Panda dataframes as images
import dataframe_image as dfi
from pandas.plotting import table

# this is for on-screen figure plotting
import matplotlib

if os.name == 'nt':
    # This works best under Windows
    matplotlib.use('Qt5Agg')
    # This import is to copy and paste figure
    # import addcopyfighandler
else:
    # This is for Linux. If ones keeps this setting also for Windows then there is memory leakage when plotting
    # figures in a loop and used memory keeps increasing at each cycle
    matplotlib.use('TKAgg')
    print('DONE!')

# set interactive mode for plotting
plt.ion()

# IMPORT DIRECTORIES
from utilities import behavior_dir, figures_dir

# save_figs == True --> figures are also saved on disk
save_figs = True

subj_files = ['subj1_behavior','subj2_behavior', 'subj3_behavior', 'subj4_behavior', 'subj5_behavior', 'subj6_behavior', 'subj7_behavior',
              'subj8_behavior', 'subj9_behavior', 'subj10_behavior', 'subj11_behavior', 'subj12_behavior']
n_subjs = len(subj_files)

# Threshold for discriminating between mirc and submirc (according to Ullman et al.'s original paper)
submirc_mirc_threshold = 0.5

# cond can be sub-mirc, mirc, full or sub-mirc post
correct_perc_df = pd.DataFrame(columns=['correct_perc', 'subj', 'cond'])
n_trials_df = pd.DataFrame(columns=['n_trials', 'subj', 'cond'])
n_correct_trials_df = pd.DataFrame(columns=['n_correct_trials', 'subj', 'cond'])

# dataframe containing the data for ALL subjects
data_all_subjs = pd.DataFrame()

for curr_subj_ind in range(0, n_subjs):
    print('Processing subject {0}'.format(curr_subj_ind+1))
    # read Excel file with the data
    f_name = os.path.join(behavior_dir, subj_files[curr_subj_ind]) + '.xlsx'
    data = pd.read_excel(f_name)

    # We presented images from 10 different categories
    # category 11 is a control condition with full, easy-to-recognize images

    # extract all presented categories and conditions
    categories = np.sort(np.unique(data['Category']))
    n_categories = len(categories)
    # In the Excel files the conditions are coded as follows:
    # 1 - submirc
    # 2 - mirc
    # 3 - mirc_2
    # 4 - full
    # 5 - submirc post
    # 6 - control
    # 7 - For some subjects there was a 7th category as "noisy" images
    conditions = np.sort(np.unique(data['Cond']))
    n_conditions = len(conditions)

    # count number of presented stimuli and correct responses
    # in each combination of category and condition
    n_tot_stimuli = np.zeros([n_conditions, n_categories])
    n_correct_stimuli = np.zeros([n_conditions, n_categories])

    # ... here we go
    for (curr_cond_ind, curr_cond) in zip(range(0, len(conditions)), conditions):
        for (curr_cat_ind, curr_cat) in zip(range(0, len(categories)), categories):
            # extract indices for current combination of condition and category
            inds = np.where(np.logical_and(data['Cond'] == curr_cond, data['Category'] == curr_cat))[0]
            # count stimuli
            n_tot_stimuli[curr_cond_ind, curr_cat_ind] = len(inds)
            if len(inds) > 0:
                # count correct responses
                n_correct_stimuli[curr_cond_ind, curr_cat_ind] = np.count_nonzero(data['CorrectResponse'][inds])

    # here we see whether we have to select a given category and, in case, where is the
    # sub-mirc/mirc boundary.
    # We presented 3 levels of degradation: sub_mirc, mirc and mirc_2
    # we can thus have that the boundary is between sub_mirc and mirc OR between mirc and mirc_2
    # If, on the other hand, we have that recognition rates for ALL THREE levels of degradation are
    # EITHER above 0.5 OR below 0.5 then
    # We code things as follows for each category
    # -1 -> DO NOT CONSIDER that category
    # 0 -> The boundary is between sub_mirc and mirc
    # 1 -> The boundary is between mirc and mirc_2
    # perc_categories = 10*np.ones(n_correct_stimuli.shape)
    # for ind_1 in range(0, perc_categories.shape[0]):
    #    for ind_2 in range(0, perc_categories.shape[0]):
    #        if n_tot_stimuli[ind_1, ind_2] != 0:
    #            perc_categories[ind_1, ind_2] =  n_correct_stimuli[ind_1, ind_2] / n_tot_stimuli[ind_1, ind_2]
    perc_categories = n_correct_stimuli / n_tot_stimuli
    # remove categories that we are not interested in
    # perc_categories = perc_categories[0:perc_categories.shape[0], 0:perc_categories.shape[1]]

    # Here I HARD-CODE that we are interested only in categories with codes 1-10
    # This is necessary as for some subjects stimuli with code=11 were presented (i.e. images with noise), while
    # for others (e.g. subj_12) fewer trials were presented.
    n_valid_categories = len(np.where(categories <= 10)[0])
    categories_recoding = -np.ones(n_valid_categories)
    for curr_cat_ind in range(0, perc_categories.shape[1]-1):
        # category that WAS NOT recognized at ANY levels
        if (perc_categories[0, curr_cat_ind] <= submirc_mirc_threshold and perc_categories[1, curr_cat_ind] <= submirc_mirc_threshold \
              and perc_categories[2, curr_cat_ind] <= submirc_mirc_threshold):
                categories_recoding[curr_cat_ind] = 10
        # category that WAS recognized at ALL levels
        elif (perc_categories[0, curr_cat_ind] > submirc_mirc_threshold and perc_categories[1, curr_cat_ind] > submirc_mirc_threshold \
          and perc_categories[2, curr_cat_ind] > submirc_mirc_threshold):
            categories_recoding[curr_cat_ind] = 9
        # boundary is between sub_mirc and mirc: NO RECODING
        elif (perc_categories[0, curr_cat_ind] <= submirc_mirc_threshold and perc_categories[1, curr_cat_ind] > submirc_mirc_threshold \
                and perc_categories[2, curr_cat_ind] > submirc_mirc_threshold):
            categories_recoding[curr_cat_ind] = 0
        # boundary is between mirc and mirc_2: RECODING
        elif (perc_categories[0, curr_cat_ind] <= submirc_mirc_threshold and perc_categories[1, curr_cat_ind] <= submirc_mirc_threshold \
                and perc_categories[2, curr_cat_ind] > submirc_mirc_threshold):
            categories_recoding[curr_cat_ind] = 1

    # pretty-print the array on screen
    np.set_printoptions(precision=2)
    print(perc_categories)

    # This is a sanity check
    if len(np.where(categories_recoding == -1)[0]) > 0:
        warnings.warn('I could not recode one category for subj -- {0}'.format(f_name))

    # Now we can recode the stimuli as follows:
    # if categories_recoding == 0 --> WE DO NOT RECODE MIRC AND SUBMIRC
    # if categories_recoding == 1 --> WE RECODE sub_mirc = 0, mirc = 1 and mirc_2 = 2 (i.e. the new sub_mirc is the original mirc)
    # if categories_recoding == 9 (recognition at ALL levels) --> WE RECODE all levels (sub_mirc, mirc and mirc_2) as 9
    # if categories_recoding == 10 (NO recognition at ANY levels) --> WE RECODE all levels (sub_mirc, mirc and mirc_2) as 10

    relabeled_cond = data['Cond'].copy()
    for curr_cat_ind in range(0, len(categories_recoding)):
        # get code for the current category
        curr_cat_code = categories[curr_cat_ind]

        if categories_recoding[curr_cat_ind] == 0:
            # We do not recode category in this case
            pass
        elif categories_recoding[curr_cat_ind] == 1:
            # WE RECODE sub_mirc = 0, mirc = 1 and mirc_2 = 2 (i.e. the new sub_mirc is the original mirc)
            inds_submirc = np.where(np.logical_and(data['Cond'] == 1, data['Category'] == curr_cat_code))[0]
            inds_mirc = np.where(np.logical_and(data['Cond'] == 2, data['Category'] == curr_cat_code))[0]
            inds_mirc_2 = np.where(np.logical_and(data['Cond'] == 3, data['Category'] == curr_cat_code))[0]
            inds_submirc_post = np.where(np.logical_and(data['Cond'] == 5, data['Category'] == curr_cat_code))[0]
            relabeled_cond[inds_submirc] = 0
            relabeled_cond[inds_mirc] = 1
            relabeled_cond[inds_mirc_2] = 2

            category_resps = data.iloc[inds_submirc_post]['CorrectResponse']
            if np.sum(category_resps) / len(category_resps) >= submirc_mirc_threshold:
                relabeled_cond[inds_submirc_post] = 8
            else:
                relabeled_cond[inds_submirc_post] = 12
        elif categories_recoding[curr_cat_ind] == 9:
            # WE RECODE all levels (sub_mirc, mirc, mirc_2 and submirc post) as 9
            inds_category = np.where(data['Category'] == curr_cat_code)[0]
            relabeled_cond[inds_category] = 9
        elif categories_recoding[curr_cat_ind] == 10:
            # Here we are processing categories that were not recognized at ANY level.
            # WE RECODE all levels (sub_mirc, mirc, mirc_2 and submirc post) as 10
            inds_category = np.where(data['Category'] == curr_cat_code)[0]
            relabeled_cond[inds_category] = 10

            # HOWEVER, if the corresponding sub-mirc is recognized above submirc_mirc_threshold (0.5)
            # then we retain that sub-mirc post category!
            # extract sub-mirc post
            inds_submirc_post = np.where(np.logical_and(data['Cond'] == 5, data['Category'] == curr_cat_code))[0]
            # recognition level of that category
            category_resps = data.iloc[inds_submirc_post]['CorrectResponse']
            if np.sum(category_resps) / len(category_resps) >= submirc_mirc_threshold:
                relabeled_cond[inds_submirc_post] = 11
        elif categories_recoding[curr_cat_ind] == -1:
            # This is the case when I could not properly recode a category. For instance, patient 3 had higher
            # recognition performances in the submirc than mirc or mirc_2 conditions. In this case, I recode everything
            # as 9 (i.e. category that was recognized at ALL levels)
            inds_category = np.where(data['Category'] == curr_cat_code)[0]
            relabeled_cond[inds_category] = 9

    # At the end of these lines we have recoded things as
    # 0 - sub-submirc (i.e. sub-mirc when recognition threshold is between mirc and mirc_2)
    # 1 - sub-mirc
    # 2 - mirc
    # 3 - mirc_2 when there is NO sub-submirc
    # 4 - full
    # 5 - sub-mirc post when there is NO sub-submirc
    # 6 - control
    # 7 - control images with noise (we do not use them)
    # 8 - sub-mirc post when there is sub-submirc and for which recognition was ABOVE the 50% threshold
    # 9 -  sub-mirc, mirc, mirc_2 and sub-mirc post for categories that were recognized at ALL levels
    # 10 - sub-mirc, mirc, mirc_2 and sub-mirc post for categories that were NOT recognized at ANY level
    # 11 - sub-mirc post for images that were recognized at NO level but were recognized in the sub-mirc post
    #       condition after exposure to the full images.
    # 12 - sub-mirc post that were category 8 ... but for which recognition was BELOW the 50% threshold
    #

    # add relabeled column to dataframe
    data.insert(loc=data.shape[1], column='RelabeledCond_Tonino', value=relabeled_cond)

    #  conditions selected for plotting
    # conditions 5 and 8 are sub-MIRC post
    selected_conditions = [1, 2, 4]

    # select only conditions in array selected_conditions above
    data_recoded = data[[tmp in selected_conditions for tmp in data['RelabeledCond_Tonino']]]

    # update selected conditions vector
    n_selected_conditions = len(selected_conditions)

    # compute categories that were valid for the current subject
    categories_recoded = np.sort(np.unique(data_recoded['Category']))
    n_categories_recoded = len(categories_recoded)

    # count number of presented stimuli and correct responses
    # in each combination of category and condition
    n_tot_stimuli_recoded = np.zeros([n_selected_conditions, n_categories_recoded])
    n_correct_stimuli_recoded = np.zeros([n_selected_conditions, n_categories_recoded])

    # compute performances with the recoded categories
    # for (curr_cond_ind, curr_cond) in zip(range(0, len(selected_conditions)), selected_conditions):
    for (curr_cond_ind, curr_cond) in enumerate(selected_conditions):
        for (curr_cat_ind, curr_cat) in enumerate(categories_recoded):
            # extract indices for current combination of condition and category
            inds = np.where(np.logical_and(data['RelabeledCond_Tonino'] == curr_cond, data['Category'] == curr_cat))[0]
            # count stimuli
            n_tot_stimuli_recoded[curr_cond_ind, curr_cat_ind] = len(inds)
            if len(inds) > 0:
                # count correct responses
                n_correct_stimuli_recoded[curr_cond_ind, curr_cat_ind] = np.count_nonzero(data['CorrectResponse'][inds])

    # percentage of correct responses in the recoded categories
    # row = 0 --- sub-mirc
    # row = 1 --- mirc
    # row = 2 --- full
    # row = 3 -- sub-mirc post
    # the following line is to prevent NaN to show up in out percentages.
    # indeed  n_correct_stimuli_recoded <= n_tot_stimuli_recoded
    # Thus if n_tot_stimuli_recoded == 0 --> n_correct_stimuli_recoded = 0
    # and n_correct_stimuli_recoded / n_tot_stimuli_recoded would be NaN
    # If we set it n_correct_stimuli_recoded = 1 then the above division will be 0
    n_tot_stimuli_recoded[n_tot_stimuli_recoded==0] = 1

    perc_categories_recoded = n_correct_stimuli_recoded / n_tot_stimuli_recoded
    perc_correct = perc_categories_recoded.mean(axis=1)

    # --------------------------------------------------------------------------------
    # ------------------Here we analyze submirc-post trials --------------------------
    # --------------------------------------------------------------------------------
    # We have to analyze them separately as for some sub-mirc-post trials we have to consider them, WITHOUT
    # however considering the corresponding mirc and sub-mirc conditions.

    # insert column coding for subject number
    data.insert(0, 'subj_num', curr_subj_ind)

    # append current dataframe
    # NOTE THAT APPEND IS PERFORMED BEFORE RECODING CONDITIONS 5 and 8!
    data_all_subjs = pd.concat([data_all_subjs, data])

    # ------------------------------------------------------------------
    # RECODE condition 8 (sub-mirc post when there is sub-submirc) and 11 (sub-mirc post when the stimulus was recognized at ALL levels)
    # as 5 (sub-mirc post when there is NO sub-submirc) to include those trials in the analysis
    # ------------------------------------------------------------------
    data.loc[data['RelabeledCond_Tonino'] == 8, 'RelabeledCond_Tonino'] = 5
    data.loc[data['RelabeledCond_Tonino'] == 11, 'RelabeledCond_Tonino'] = 5
    # data.loc[data['RelabeledCond_Tonino'] == 12, 'RelabeledCond_Tonino'] = 5

    # select only conditions in array selected_conditions above
    data_recoded_submirc_post = data.loc[data['RelabeledCond_Tonino'] == 5]

    # compute categories that were valid for the current subject
    categories_recoded_submirc_post = np.sort(np.unique(data_recoded_submirc_post['Category']))
    n_categories_recoded_submirc_post = len(categories_recoded_submirc_post)

    # count number of presented stimuli and correct responses
    # in each combination of category and condition
    n_tot_stimuli_recoded_submirc_post = np.zeros([1, n_categories_recoded_submirc_post])
    n_correct_stimuli_recoded_submirc_post = np.zeros([1, n_categories_recoded_submirc_post])

    # compute performances with the recoded categories
    # for (curr_cond_ind, curr_cond) in zip(range(0, len(selected_conditions)), selected_conditions):
    for (curr_cat_submirc_post_ind, curr_cat_submirc_post) in enumerate(categories_recoded_submirc_post):
        # extract indices for current combination of condition and category
        inds = np.where(np.logical_and(data['RelabeledCond_Tonino'] == 5, data['Category'] == curr_cat_submirc_post))[0]
        # count stimuli
        n_tot_stimuli_recoded_submirc_post[0, curr_cat_submirc_post_ind] = len(inds)
        if len(inds) > 0:
            # count correct responses
            n_correct_stimuli_recoded_submirc_post[0, curr_cat_submirc_post_ind] = np.count_nonzero(data['CorrectResponse'][inds])

    perc_categories_recoded_submirc_post = n_correct_stimuli_recoded_submirc_post / n_tot_stimuli_recoded_submirc_post
    perc_correct_submirc_post = perc_categories_recoded_submirc_post.mean(axis=1)
    print('subj: {0} --- n_stimuli: {1} --- n_correct: {2}'.format(curr_subj_ind, n_tot_stimuli_recoded_submirc_post, n_correct_stimuli_recoded_submirc_post))

    # ---------------------------------------------------------------------------
    # ------------------- I am done with computations -----------------------
    # join results for all conditions and sub_mirc_post
    perc_correct_all = np.concatenate((perc_correct, perc_correct_submirc_post))

    #               add rows in dataframe
    correct_perc_df = pd.concat([correct_perc_df,
                                 pd.DataFrame({'correct_perc':perc_correct_all,
                                               'subj':len(perc_correct_all)*[curr_subj_ind],
                                               'cond': ['submirc', 'mirc', 'full', 'submirc_post']})], ignore_index = True)

    # dataframe for the total number of trials
    n_trials_all = np.concatenate((np.sum(n_tot_stimuli_recoded, axis=1), np.sum(n_tot_stimuli_recoded_submirc_post, axis=1)))
    n_trials_df = pd.concat([n_trials_df,
                             pd.DataFrame({'n_trials': n_trials_all,
                                           'subj': len(n_trials_all) * [curr_subj_ind],
                                           'cond': ['submirc', 'mirc', 'full', 'submirc_post']})], ignore_index=True)

    # dataframe for the number of CORRECT trials
    n_correct_all = np.concatenate((np.sum(n_correct_stimuli_recoded, axis=1), np.sum(n_correct_stimuli_recoded_submirc_post, axis=1)))
    n_correct_trials_df = pd.concat([n_correct_trials_df,
                                 pd.DataFrame({'n_correct_trials':n_correct_all,
                                               'subj':len(n_correct_all)*[curr_subj_ind],
                                               'cond': ['submirc', 'mirc', 'full', 'submirc_post']})], ignore_index = True)

    # bring back the RelabeledCond_Tonino column to its original values
    data['RelabeledCond_Tonino'] = relabeled_cond

    # save data_frame with a different name
    f_name = os.path.join(behavior_dir, subj_files[curr_subj_ind]) + '_Tonino.xlsx'
    data.to_excel(f_name)

# We are done with the analysis
print('DONE')

# now we parse the dataframe containing data for ALL subjects to obtain the number of trials for each condition
# and subject USED for the data analysis
conds_codes = ((1,), (2,), (4,), (5, 8, 11))
conds_names = ('submirc', 'mirc', 'full', 'submirc_post')

n_conds = len(conds_codes)
n_trials_for_analysis_df = pd.DataFrame(columns=conds_names)
for curr_subj_ind in range(0, n_subjs):
    tmp_row = np.zeros(n_conds)
    # extract dataframe for current subject
    curr_df = data_all_subjs[data_all_subjs['subj_num']==curr_subj_ind]
    for curr_cond_ind, curr_cond_name in enumerate(conds_names):
        # extract number of trials in the given condition
        cond_trial_inds = [inds for (inds, val) in enumerate(curr_df['RelabeledCond_Tonino']) if val in conds_codes[curr_cond_ind]]
        tmp_row[curr_cond_ind] = len(cond_trial_inds)
    # append results to dataframe
    n_trials_for_analysis_df.loc[len(n_trials_for_analysis_df)] = tmp_row


# -------------------- correct_perc_df: transform data frames for pretty-printing --------------------
cond_names = np.unique(correct_perc_df['cond'])
field_names = np.insert(cond_names, 0, 'subj')
subj_names = np.unique(correct_perc_df['subj'])
# create dataframe
correct_perc_by_subj_df = pd.DataFrame(columns=field_names)
n_trials_by_subj_df = pd.DataFrame(columns=field_names)
n_correct_trials_by_subj_df = pd.DataFrame(columns=field_names)

for curr_subj in subj_names:
    # correct_perc_df: extract values for the current subject
    tmp = ['subj{0}'.format(curr_subj+1)]
    for curr_cond_ind, curr_cond in enumerate(cond_names):
        tmp.append(float((correct_perc_df.loc[(correct_perc_df['subj']==curr_subj) & (correct_perc_df['cond']==curr_cond)]['correct_perc']).iloc[0]))
    # now append row in the data frame
    correct_perc_by_subj_df.loc[len(correct_perc_by_subj_df)] = tmp
    # ------------------------------------------------------------------
    # n_trials_df: extract values for the current subject
    tmp = ['subj{0}'.format(curr_subj+1)]
    for curr_cond_ind, curr_cond in enumerate(cond_names):
        tmp.append(float((n_trials_df.loc[(n_trials_df['subj']==curr_subj) & (n_trials_df['cond']==curr_cond)]['n_trials']).iloc[0]))
    # now append row in the data frame
    n_trials_by_subj_df.loc[len(n_trials_by_subj_df)] = tmp
    # ------------------------------------------------------------------
    # n_correct_trials_df: extract values for the current subject
    tmp = ['subj{0}'.format(curr_subj+1)]
    for curr_cond_ind, curr_cond in enumerate(cond_names):
        tmp.append(float((n_correct_trials_df.loc[(n_correct_trials_df['subj']==curr_subj) & (n_correct_trials_df['cond']==curr_cond)]['n_correct_trials']).iloc[0]))
    # now append row in the data frame
    n_correct_trials_by_subj_df.loc[len(n_correct_trials_by_subj_df)] = tmp

# save tables with number of trials IN PNG format, number of correct trials and percentages of correct trials as figures
if save_figs == True:
    # ------------------ save dataframe n_trials_by_subj_df as PNG image ------------------
    f_name = os.path.join(figures_dir, 'table_trials_per_condition')
    axes = plt.subplot(frame_on=False)
    # hide axes
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)

    axes.table(cellText=n_trials_by_subj_df.values, colLabels=n_trials_by_subj_df.keys(), loc='center')
    plt.tight_layout()
    plt.savefig(fname=f_name + '.png', dpi=300)

# compute average and std and plot results
avgs = correct_perc_df.groupby('cond')['correct_perc'].mean().values
cond_names = np.unique(correct_perc_df['cond'])
stds = correct_perc_df.groupby('cond')['correct_perc'].std().values

# order in which the bars are plotted
conds_order = ['submirc', 'mirc', 'full', 'submirc_post']
# labels used in the figures for the different conditions
conds_labels = ['sub-MIRC', 'MIRC', 'object', 'sub-MIRC\npost']
conds_colors = ['red', 'cornflowerblue', 'black', 'red']
conds_hatches = ['', '', '', '/'*3]
conds_fills = [True, True, True, False]
inds = [np.where(cond_names == curr_cond)[0][0] for curr_cond in conds_order]

# Plot behavioral performances as BARPLOT
# set width of the lines used for the barplot fill (https://stackoverflow.com/questions/29549530/how-to-change-the-linewidth-of-hatch-in-matplotlib)
matplotlib.rcParams['hatch.linewidth'] = 2
fig, axs = plt.subplots(figsize=(6, 6))
axs.bar(x=conds_labels, height=avgs[inds], fill=True, edgecolor='white', color=conds_colors, hatch=conds_hatches)
axs.errorbar(x=conds_labels, y=avgs[inds], yerr=stds[inds] / np.sqrt(n_subjs), fmt='none', elinewidth=3, capsize=3, color='black')
axs.set_ylabel('fraction correct', fontsize=24)
axs.set_ylim(0, 1)
axs.tick_params(axis='y', labelsize=22)
axs.tick_params(axis='x', labelsize=18)

# we want an "open box" plot
axs.spines['right'].set_color('none')
axs.spines['top'].set_color('none')
plt.tight_layout()

# save the plot
if save_figs == True:
    f_name = os.path.join(figures_dir, 'MIRCs_Behavior')
    fig.savefig(fname=f_name + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(fname=f_name + '.svg', dpi=300, bbox_inches='tight')
    # fig.savefig(fname=f_name + '.eps', dpi=1000, bbox_inches='tight')

# print average performances of MIRC, sub-MIRC and sub-MIRC post
print()
print('AVG\tMIRC\tsub-MIRC\tsub-MIRC post')
print('\t{0:1.3f}\t{1:1.3f}\t{2:1.3f}'.format(np.mean(correct_perc_by_subj_df['mirc']), np.mean(correct_perc_by_subj_df['submirc']),\
                                  np.mean(correct_perc_by_subj_df['submirc_post'])))


print('DONE!')

# perform statistical tests
print('Statistical comparison between MIRC and SUBMIRC performances')
print(stats.mannwhitneyu(correct_perc_by_subj_df['mirc'], correct_perc_by_subj_df['submirc']))
print(stats.wilcoxon(correct_perc_by_subj_df['mirc'], correct_perc_by_subj_df['submirc']))
print(stats.ttest_rel(correct_perc_by_subj_df['mirc'], correct_perc_by_subj_df['submirc']))
print()
print('Statistical comparison between SUBMIRC post and SUBMIRC performances')
print(stats.mannwhitneyu(correct_perc_by_subj_df['submirc_post'], correct_perc_by_subj_df['submirc']))
print(stats.wilcoxon(correct_perc_by_subj_df['submirc_post'], correct_perc_by_subj_df['submirc']))
print(stats.ttest_rel(correct_perc_by_subj_df['submirc_post'], correct_perc_by_subj_df['submirc']))

# compare results by age group
ages = [17, 25, 18, 15, 35, 26, 12, 43, 22, 11, 21, 12]
correct_perc_by_subj_df['age'] = ages

# perform MIRC comparisons between age groups
group_1 = correct_perc_by_subj_df['mirc'][correct_perc_by_subj_df['age']<25]
group_2 = correct_perc_by_subj_df['mirc'][correct_perc_by_subj_df['age']>=25]
print('Comparing MIRC recognition performances: group 1 (<25 years old, n={0}) vs group 2 (>=25 years old, n={1})'.\
      format(len(group_1), len(group_2)))
print(stats.mannwhitneyu(group_1, group_2))

# perform sub-MIRC post comparisons between age groups
group_1 = correct_perc_by_subj_df['submirc_post'][correct_perc_by_subj_df['age']<25]
group_2 = correct_perc_by_subj_df['submirc_post'][correct_perc_by_subj_df['age']>=25]
print('Comparing sub-MIRC post recognition performances: group 1 (<25 years old, n={0}) vs group 2 (>=25 years old, n={1})'.\
      format(len(group_1), len(group_2)))
print(stats.mannwhitneyu(group_1, group_2))

# perform sub-MIRC comparisons between age groups
group_1 = correct_perc_by_subj_df['submirc'][correct_perc_by_subj_df['age']<25]
group_2 = correct_perc_by_subj_df['submirc'][correct_perc_by_subj_df['age']>=25]
print('Comparing sub-MIRC recognition performances: group 1 (<25 years old, n={0}) vs group 2 (>=25 years old, n={1})'.\
      format(len(group_1), len(group_2)))
print(stats.mannwhitneyu(group_1, group_2))

print('DONE!')

