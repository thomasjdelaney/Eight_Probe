"""
For varying the bin width used from 0.005 to 2 seconds, and taking measurements using these bin widths.

We save a frame for spike counts for each time bin containing all cells. We also save a frame for each time bin for all pairs.
"""
import os, argparse, sys, shutil
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product, combinations
from scoop import futures
from multiprocessing import Pool
from functools import reduce

parser = argparse.ArgumentParser(description='For varying the bin width used from 0.005 to 4 seconds, and taking measurements using these bin widths.')
parser.add_argument('-n', '--number_of_cells', help='Number of cells to process. Use 0 for all.', type=int, default=10)
parser.add_argument('-f', '--save_firing_rate_frame', help='Flag to indicate whether or not firing rates should be saved.', default=False, action='store_true')
parser.add_argument('-a', '--save_analysis_frame', help='Flag to indicate whether or not analysis should be performed and saved.', default=False, action='store_true')
parser.add_argument('-z', '--save_conditional_correlations', help='Flag to indicate whether or not to calculate and save conditional correlations.', default=False, action='store_true')
parser.add_argument('-c', '--num_chunks', help='Number of chunks to split the pairs into before processing.', default=10, type=int)
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows',30) # setting display options for terminal display
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

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
    Returns:    svd_marginal_distn, P(Z_i = z_i)
                conditional_expectation_dict, cell_id => conditional expectation of the spike counts given the svd comp, numpy array (float) of length = num_bins_svd

    NB check this again.
    """
    svd_counts, svd_bins = np.histogram(svd_comp, bins=num_bins_svd)
    svd_marginal_distn = svd_counts / svd_counts.sum()
    conditional_expectation_dict = {}
    for cell_id, spike_counts in spike_count_dict.items():
        spike_count_list = list(range(spike_counts.min(), spike_counts.max()+1)) # list for faster indexing
        joint_distn = np.zeros((len(spike_count_list), num_bins_svd), dtype=float) 
        for i,(svd_bin_start, svd_bin_stop) in enumerate(zip(svd_bins[:-1], svd_bins[1:])):
            svd_bin_value_times = svd_times[np.logical_and(svd_bin_start <= svd_comp, svd_comp < svd_bin_stop)]
            if svd_bin_value_times.size > 0:
                svd_bin_value_time_bin_inds = np.digitize(svd_bin_value_times, time_bins)
                svd_bin_value_spike_count_values, svd_bin_value_spike_count_counts = np.unique(spike_counts[svd_bin_value_time_bin_inds-1], return_counts=True)
                joint_distn[[spike_count_list.index(spikes) for spikes in svd_bin_value_spike_count_values], i] += svd_bin_value_spike_count_counts 
        joint_distn = joint_distn / joint_distn.sum()
        cond_distn = joint_distn / svd_marginal_distn
        cond_distn[np.isnan(cond_distn)] = 0.0
        conditional_expectation_dict[cell_id] = np.zeros(num_bins_svd, dtype=float)
        for i,sc in enumerate(spike_count_list):
            conditional_expectation_dict[cell_id] += sc * cond_distn[i]
    return svd_marginal_distn, conditional_expectation_dict

def calcWeightedProductCondExp(svd_marginal_dist, first_cond_exp, second_cond_exp):
    """
    For calculating the weighted product of two conditional expectations.
    Arguments:  svd_marginal_dist, numpy array (float), the marginal distribution of the singular value decomposition to use.
                first_cond_exp, numpy array (float), first conditional expectation.
                second_cond_exp, numpy array (float), second conditional expectation.
    Return:     float E[E[X|Z] * E[Y|Z]]
    """
    return np.dot(svd_marginal_dist, first_cond_exp * second_cond_exp)

def reduceWeightedProductCondExpFuture(comp_contribution, exp_prod_cond_exp):
    """
    For adding the expected values of the product conditional expectations to the conditional sum array.
    Arguments:  comp_contribution, numpy array (float), initially nans, add to each element incrementally
                exp_prod_cond_exp, float
    Returns:    numpy array (float)
    """
    first_nan_ind = next(i for i,x in enumerate(comp_contribution.flatten()) if np.isnan(x))
    first_nan_ind = np.unravel_index(first_nan_ind, comp_contribution.shape)
    comp_contribution[first_nan_ind] = exp_prod_cond_exp
    return comp_contribution

def getExpCondCov(mouse_face, spike_count_dict, time_bins, num_bins_svd=50):
    """
    For calculating the expected value of the conditional covariance between spike counts.
    Arguments:  mouse_face, dict, contains all info about the mouse films,
                spike_count_dict, Dict, cell_id => spike counts
                time_bins, the spike count time bin borders,
    Returns:    E[cov(cell_1, cell_2 | Z_1, ..., Z_500)] expected covariance

    NB getting negative expected variance at the moment, problem unknown.
    """
    num_cells = len(spike_count_dict)
    num_comps = mouse_face.get('motionSVD').shape[1]
    cell_ids = list(spike_count_dict.keys())
    spike_count_array = np.array(list(spike_count_dict.values()))
    svd_times, svd_comps = ep.getRelevantMotionSVD(mouse_face, time_bins)
    with Pool() as pool:
        cond_exp_futures = pool.starmap_async(getConditionalExpectation, zip(500*[spike_count_dict], 500*[time_bins], svd_comps.T, 500*[svd_times]))
        cond_exp_futures.wait()
    cond_exp_got = cond_exp_futures.get()
    conditional_sum = np.zeros((num_cells, num_cells), dtype=float)
    # conditional_log_sum = np.zeros((num_cells, num_cells), dtype=float)
    for c in range(num_comps):
        svd_marginal_dist, cond_exp_dict = cond_exp_got[c]
        for i,j in combinations(range(num_cells), 2):
            conditional_sum[i,j] += np.dot(svd_marginal_dist, cond_exp_dict[cell_ids[i]] * cond_exp_dict[cell_ids[j]])
            # conditional_log_sum[i,j] += np.log(np.dot(svd_marginal_dist, cond_exp_dict[cell_ids[i]] * cond_exp_dict[cell_ids[j]]))
    conditional_sum = conditional_sum + conditional_sum.T
    # conditional_log_sum = conditional_log_sum + conditional_log_sum.T
    for c in range(num_comps):
        svd_marginal_dist, cond_exp_dict = cond_exp_got[c]
        for i in range(num_cells):
            conditional_sum[i,i] += np.dot(svd_marginal_dist, cond_exp_dict[cell_ids[i]] * cond_exp_dict[cell_ids[i]])
            # conditional_log_sum[i,i] += np.log(np.dot(svd_marginal_dist, cond_exp_dict[cell_ids[i]] * cond_exp_dict[cell_ids[i]]))
    mean_of_products_of_spike_counts = np.array([np.outer(s, s) for s in spike_count_array.T]).mean(axis=0)
    product_of_mean_spike_counts = np.outer(spike_count_array.mean(axis=1), spike_count_array.mean(axis=1))
    return mean_of_products_of_spike_counts + ((num_comps-1) * product_of_mean_spike_counts) - conditional_sum
    # return mean_of_products_of_spike_counts - np.exp(conditional_log_sum - ((num_comps-1) * np.log(product_of_mean_spike_counts)))

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

