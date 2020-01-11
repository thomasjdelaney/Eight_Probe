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
parser.add_argument('-z', '--save_conditional_correlations', help='Flag to indicate whether or not to calculate and save conditional correlations.', default=False, action='store_true')
parser.add_argument('-c', '--num_chunks', help='Number of chunks to split the pairs into before processing.', default=10, type=int)
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

def constructMapFuncArgs(pairs, spike_count_dict):
    """
    For constructing a list of dictionaries to be passed into a mapping function for scoop.futures.mapReduce.
    In this context the mapping function can only take one argument.
    Arguments:  pairs, numpy.array, all the possible pairs.
                spike_count_dict, cell_id => spike counts
    Returns:    List of dictionaries, each with two keys, values are arrays of spike counts.
    """
    dict_list = list([])
    for pair in pairs:
        dict_list.append({pair[0]:spike_count_dict[pair[0]], pair[1]:spike_count_dict[pair[1]]})
    return dict_list

def constructMapFuncArgsOld(pairs, spike_count_frame):
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

def getConditionalExpectation(spike_count_dict, time_bins, svd_comp, svd_times, num_bins_svd=50):
    """
    For calculating the conditional expectation of the spike counts given num_bins_svd different binned values of svd_comp.
    Arguments:  spike_count_dict, dict cell_id => spike_counts
                time_bins, numpy array (float), times for the spike counts.
                svd_comp, numpy array (float), singular value decomposition component
                svd_times, numpy array (float), times for the svd measurements
    Returns:    conditional expectation of the spike counts given the svd comp, numpy array (float) of length = num_bins_svd
    """
    svd_counts, svd_bins = np.histogram(svd_comp, bins=num_bins_svd)
    spike_count_values = np.arange(spike_counts.min(), spike_counts.max()+1)
    svd_marginal_distn = svd_counts / svd_counts.sum()
    joint_distn_dict = dict(zip(spike_count_dict.keys(), np.zeros((len(spike_count_dict), num_bins_svd, spike_count_values.size), dtype=int)))
    for cell_id, spike_counts in spike_count_dict:
        for i,(svd_bin_start, svd_bin_stop) in enumerate(zip(svd_bins[:-1], svd_bins[1:])):
            svd_bin_value_times = svd_times[np.logical_and(svd_bin_start <= svd_comp, svd_comp < svd_bin_stop)]
            svd_bin_value_time_bin_inds = np.digitize(svd_bin_value_times, time_bins)
            svd_bin_value_spike_count_values, svd_bin_value_spike_count_counts = np.unique(spike_counts[svd_bin_value_time_bin_inds-1], return_counts=True)
            joint_distn_dict[cell_id][i, svd_bin_value_spike_count_values] += svd_bin_value_spike_count_counts
            # can put a loop in here to run through each cell. 
            # better to return a dictionary similar to spike_count_dict, but for conditional expectations

def reduceAnalysisDicts(first_dict, second_dict):
    """
    Each dict is str => numpy array, and will have the same keys. This function appends the dictionaries together. This is the reduce function for mapReduce.
    Arguments:  first_dict, keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
                second_dict, keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
    Return:     Dict, keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
    """
    return {k:np.append(first_dict[k], second_dict[k]) for k in first_dict.keys()}

def saveAnalysisFrame(analysis_frame, chunk_num, save_file):
    """
    Saves the analysis_frame to save_file as a csv with or without a header according to chunk_num. 
    If chunk_num == 0, save with header, else append.
    """
    if chunk_num == 0:
        analysis_frame.to_csv(save_file, index=False)
    else:
        analysis_frame.to_csv(save_file, mode='a', header=False, index=False)
    return None

if (not args.debug) & (__name__ == "__main__"):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for m,mouse_name in enumerate(ep.mouse_names):
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        spon_start_time = ep.spon_start_times[m]
        cell_ids = cell_info[cell_info.mouse_name == mouse_name].index.values
        cell_ids = ep.getRegionallyDistributedCells(cell_info.loc[cell_info.mouse_name == mouse_name], args.number_of_cells)
        spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
        pairs = np.array(list(combinations(cell_ids, 2)))
        chunked_pairs = np.array_split(pairs, args.num_chunks)
        for bin_width in ep.selected_bin_widths:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
            spike_count_dict = ep.getSpikeCountDict(spike_time_dict, bin_width, spon_start_time)
            if args.save_firing_rate_frame:
                firing_rate_frame = ep.getFiringRateFrameFromSpikeCountDict(spike_count_dict, bin_width)
                save_file = os.path.join(npy_dir, 'firing_rate_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'firing.npy')
                firing_rate_frame.to_pickle(save_file)
                print(dt.datetime.now().isoformat() + ' INFO: ' + save_file + ' saved.')
            if args.save_analysis_frame:
                save_file = os.path.join(csv_dir, 'analysis_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'analysis.csv')
                removed = os.remove(save_file) if os.path.exists(save_file) else None
                for i,pair_chunk in enumerate(chunked_pairs):
                    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing chunk number ' + str(i) + '...')
                    analysis_frame = pd.DataFrame.from_dict(futures.mapReduce(getAnalysisDictForPair, reduceAnalysisDicts, constructMapFuncArgs(pair_chunk, spike_count_dict)))
                    analysis_frame['bin_width'] = bin_width
                    saveAnalysisFrame(analysis_frame, i, save_file)
                print(dt.datetime.now().isoformat() + ' INFO: ' + save_file + ' saved.')
            if args.save_conditional_correlations:
                print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing conditional correlations...')
                mouse_face = ep.loadVideoDataForMouse(mouse_name, mat_dir)
                mouse_face = ep.getSpikeCountHistsForMotionSVD(mouse_face, spike_count_dict, ep.getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time))
                mouse_face = ep.getMouseFaceCondSpikeCounts(mouse_face, spike_time_dict)
                #for i,pair_chunk in enumerate(chunked_pairs):
                
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

