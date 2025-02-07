'''
This file contains routines that are used in the paper:

Neural correlates of minimal recognizable configurations in the human brain
by Casile et al.

author:
Antonino Casile
University of Messina
antonino.casile@unime.it
toninocasile@gmail.com

'''
import os
import numpy as np
from parse import parse
from collections import namedtuple

# ----------------------------- IMPORTANT -----------------------------
# SET THIS BASE DIRECTORY TO WHEREVER YOU UNZIPPED THE DOWNLOADED FILES
base_dir = ''

behavior_dir = os.path.join(base_dir, 'patients_data', 'behavior')
code_dir = os.path.join(base_dir, 'code')
ifps_base_dir = os.path.join(base_dir, 'patients_data', 'IFP_recordings')
electrodes_location_dir = os.path.join(base_dir, 'patients_data', 'electrode locations')
# figures_dir = os.path.join(base_dir, 'figures')
figures_dir = base_dir
# results_dir = os.path.join(base_dir, 'results')
results_dir = base_dir
patients_info_file = os.path.join(base_dir, 'patients_data', 'PatientInfo.xlsx')

# declare named tuple to return results of the responsivity and selectivity analyses
res_analysis = namedtuple('res_analysis', ['n_chans', 'chans_name', \
            'time_begin_ms', 'time_end_ms', 'chans_pos', 'chans_area', 'chans_macro_area', 'chans_hemisphere'])

'''
This functions opens and parse a text file containing the results of my analyses.
Each line of these files is in the form:
subj1_ch_002	482.00	549.00	F	-11.183505	42.445179	-24.447437	G_orbital	1
where the last number can be 1, 0, and -1.
-1 --> this electrode could not be assigned to a specific area/region/hemisphere
1 --> left hemisphere
0 --> right hemisphere
The other parameters represent the start and end of significativity/selectivity (depending
on the analysis)
The macro-area (F=frontal, P=parietal, T=temporal and O=occipital)
The following three numbers are the position and the string is the region
----
CALL:

(electrodes_names, electrodes_time_begin_ms, electrodes_time_end_ms, electrodes_region, electrodes_pos, electrodes_area, \
        electrodes_subj, n_chans_with_no_macro_area) = \
    load_results(f_name, return_only_located)

INPUTS:
    f_name = name of the text file containing the results
    return_only_located = if True --> returns only electrodes for which a brain position could be determined
OUTPUTS:
    electrodes_names = list containing the names of the electrodes
    electrodes_time_begin_ms, electrodes_time_end_ms = arrays containing, for each electrode, the start and end
        of the responsivity/selectivity interval
    electrodes_pos = positions of the electrodes
    electrodes_region = macro-area where the electrode is located (i.e. temporal, parietal, occipital or frontal)
    
'''
def load_results(f_name, return_only_located = False):
    # open result file and read lines
    fid = open(f_name)
    data = fid.readlines()
    fid.close()
    # total number of significant electrodes
    n_significant_els = len(data)

    # allocate matrix containing the position of significant electrodes
    electrodes_names = n_significant_els * [None]
    electrodes_pos = np.nan * np.ones((n_significant_els, 3))
    electrodes_time_begin_ms = np.zeros(n_significant_els)
    electrodes_time_end_ms = np.zeros(n_significant_els)
    electrodes_region = n_significant_els * [None]
    electrodes_area = n_significant_els * [None]
    electrodes_subj = np.zeros(n_significant_els)
    is_valid_electrode = np.zeros(n_significant_els).astype(bool)

    # now parse the file one line at a time
    n_chans_with_no_macro_area = 0
    for curr_electrode_ind in range(0, n_significant_els):
        # parse current line
        strs = parse('{}_ch_{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n', data[curr_electrode_ind])
        curr_subj_name = strs[0]
        curr_subj_num = int(parse('subj{}', curr_subj_name)[0])
        curr_electrode_num = int(strs[1])
        curr_time_begin_ms = float(strs[2])
        curr_time_end_ms = float(strs[3])
        curr_electrode_macro_area = strs[4]
        curr_electrode_pos = np.zeros(3)
        curr_electrode_pos[0] = float(strs[5])
        curr_electrode_pos[1] = float(strs[6])
        curr_electrode_pos[2] = float(strs[7])
        curr_electrode_area = strs[8]
        curr_electrode_hemisphere = int(strs[9])

        # check if electrode was among those with brain localizations
        electrodes_time_begin_ms[curr_electrode_ind] = curr_time_begin_ms
        electrodes_time_end_ms[curr_electrode_ind] = curr_time_end_ms
        electrodes_region[curr_electrode_ind] = curr_electrode_macro_area
        electrodes_subj[curr_electrode_ind] = curr_subj_num
        electrodes_pos[curr_electrode_ind, : ] = curr_electrode_pos
        electrodes_names[curr_electrode_ind] = '{0}_ch_{1:03d}'.format(curr_subj_name, curr_electrode_num)
        electrodes_area[curr_electrode_ind] = curr_electrode_area

        if curr_electrode_macro_area in ('NA', 'NA_POS'):
            # print('Electrode {0}_ch_{1} NOT LOCATED'.format(curr_subj_name, curr_electrode_num))
            n_chans_with_no_macro_area = n_chans_with_no_macro_area + 1
        else:
            is_valid_electrode[curr_electrode_ind] = True

    if return_only_located == True:
        # remove elements that could not be assigned a brain position
        electrodes_time_begin_ms = electrodes_time_begin_ms[is_valid_electrode]
        electrodes_time_end_ms = electrodes_time_end_ms[is_valid_electrode]
        electrodes_region = [curr_val for (curr_val, curr_is_valid) in zip(electrodes_region, is_valid_electrode) if curr_is_valid == True]
        electrodes_subj = electrodes_subj[is_valid_electrode]
        electrodes_pos = electrodes_pos[is_valid_electrode, :]
        electrodes_names = [curr_val for (curr_val, curr_is_valid) in zip(electrodes_names, is_valid_electrode) if curr_is_valid == True]
        electrodes_area = [curr_val for (curr_val, curr_is_valid) in zip(electrodes_area, is_valid_electrode) if curr_is_valid == True]

    return electrodes_names, electrodes_time_begin_ms, electrodes_time_end_ms, electrodes_region, electrodes_pos, electrodes_area, \
        electrodes_subj, n_chans_with_no_macro_area



'''
This routine returns the bad channels for each subject. That is channels strongly corrupted by noise
'''
def get_bad_channels(subj_num):
    if (subj_num < 1) or (subj_num > 12):
        raise Warning('Subject number {0} is NOT recognized'.format(subj_num))

    if subj_num == 1:
        bad_channels = [12, 13, 14, 178, 235, 237]
    elif subj_num == 2:
        bad_channels = [122] + [131, 132, 133, 134, 135, 136] + [152, 153, 182]
    elif subj_num == 4:
        bad_channels = list(range(2, 213))
    elif subj_num == 5:
        bad_channels = [87, 133, 134, 135]
    elif subj_num == 6:
        bad_channels = [114]
    elif subj_num == 10:
        bad_channels = [72, 73, 80, 81, 118, 119, 120, 121, 122, 123, 124, 125, 146, 147, 148, 149, 150, 151,\
                        152, 153, 160, 161, 162, 163, 164, 165, 166, 167, 168]
    else:
        bad_channels = []

    return bad_channels

