"""
For varying the bin width used from 0.005 to 2 seconds, and taking measurements using these bin widths.

We save a frame for spike counts for each time bin containing all cells. We also save a frame for each time bin for all pairs.
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product, combinations
from scoop import futures

parser = argparse.ArgumentParser(description='For varying the bin width used from 0.005 to 4 seconds, and taking measurements using these bin widths.')
parser.add_argument('-n', '--number_of_cells', help='Number of cells to process. Use 0 for all.', type=int, default=10)
parser.add_argument('-f', '--save_firing_rate_frame', help='Flag to indicate whether or not firing rates should be saved.', default=False, action='store_true')
parser.add_argument('-a', '--save_analysis_frame', help='Flag to indicate whether or not analysis should be performed and saved.', default=False, action='store_true')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows',30) # setting display options for terminal display

proj_dir = os.path.join(os.environ['PROJ'], 'Eight_Probe')
py_dir = os.path.join(proj_dir, 'py')
npy_dir = os.path.join(proj_dir, 'npy')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images')
mat_dir = os.path.join(proj_dir, 'mat')

sys.path.append(os.environ['PROJ'])
import Eight_Probe.py as ep

def constructMapFuncArgs(pairs, spike_count_frame):
    """
    For constructing a list of dictionaries to be passed into a mapping function for scoop.futures.mapReduce. 
    In this context the mapping function can only take one argument.
    Arguments:  pairs, numpy.array, all the possible pairs.
                spike_count_frame, DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time 
    """
    return [{pair[0]:spike_count_frame.loc[spike_count_frame.cell_id == pair[0], 'spike_count'].values, pair[1]:spike_count_frame.loc[spike_count_frame.cell_id == pair[1], 'spike_count'].values} for pair in pairs]

def saveSpikeCountFrame(cell_ids, bin_width, spike_time_dict, spon_start_time, mouse_name):
    """
    For getting, saving, and returning the spike count frame, with the bin width column. Also returns the name of the file where the spike_count_bins was saved.
    """
    spike_count_bins = ep.getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
    spike_count_frame = ep.getSpikeCountBinFrame(cell_ids, spike_time_dict, spike_count_bins)
    spike_count_frame['bin_width'] = bin_width
    save_file = os.path.join(npy_dir, 'spike_count_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'spike_counts.npy')
    spike_count_frame.to_pickle(save_file)
    return save_file, spike_count_frame

def getAnalysisDictForPair(pair_count_dict):
    """
    For getting a dictionary containing measurements for the given pair. This function is most useful for parallel processing
    as there will be a great number of pairs. This is the mapping function for mapReduce.
    Arguments:  pair_count_dict, int => numpy array int, the two keys is the pair, the values are the spike counts
    Returns:    Dict,   keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv, bias_corrected_mi)
    """
    pair = np.array(list(pair_count_dict.keys()))
    corr, corr_pv, shuff_corr, shuff_corr_pv = ep.getSpikeCountCorrelationsForPair(pair_count_dict)
    plugin_mi, plugin_shuff_mi, bias_corrected_mi = ep.getMutualInfoForPair(pair_count_dict)
    return {'corr_coef': np.repeat(corr,1), 'corr_pv':np.repeat(corr_pv,1), 'first_cell_id':np.repeat(pair[0],1), 'plugin_mi':np.repeat(plugin_mi,1), 'plugin_shuff_mi':np.repeat(plugin_shuff_mi,1), 'second_cell_id':np.repeat(pair[1],1), 'shuff_corr':np.repeat(shuff_corr,1), 'shuff_corr_pv':np.repeat(shuff_corr_pv,1), 'bias_corrected_mi':np.repeat(bias_corrected_mi,1)}

def reduceAnalysisDicts(first_dict, second_dict):
    """
    Each dict is str => numpy array, and will have the same keys. This function appends the dictionaries together. This is the reduce function for mapReduce.
    Arguments:  first_dict, keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
                second_dict, keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
    Return:     Dict, keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
    """
    return {k:np.append(first_dict[k], second_dict[k]) for k in first_dict.keys()}

if (not args.debug) & (__name__ == "__main__"):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for m,mouse_name in enumerate(ep.mouse_names):
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        spon_start_time = ep.spon_start_times[m]
        cell_ids = cell_info[cell_info.mouse_name == mouse_name].index.values
        cell_ids = cell_ids[:args.number_of_cells] if args.number_of_cells > 0 else cell_ids # selecting fewer cells for testing
        spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
        pairs = np.array(list(combinations(cell_ids, 2)))
        for bin_width in ep.selected_bin_widths:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
            spike_count_save_file, spike_count_frame = saveSpikeCountFrame(cell_ids, bin_width, spike_time_dict, spon_start_time, mouse_name)
            if args.save_firing_rate_frame:
                firing_rate_frame = ep.getFiringRateFrameFromSpikeCountFrame(spike_count_frame, bin_width)
                save_file = os.path.join(npy_dir, 'firing_rate_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'firing.npy')
                firing_rate_frame.to_pickle(save_file)
                print(dt.datetime.now().isoformat() + ' INFO: ' + save_file + ' saved.')
            if args.save_analysis_frame:
                analysis_frame = pd.DataFrame.from_dict(futures.mapReduce(getAnalysisDictForPair, reduceAnalysisDicts, constructMapFuncArgs(pairs, spike_count_frame)))
                analysis_frame['bin_width'] = bin_width
                save_file = os.path.join(npy_dir, 'analysis_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'analysis.npy')
                analysis_frame.to_pickle(save_file)
                print(dt.datetime.now().isoformat() + ' INFO: ' + save_file + ' saved.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

