import os, sys, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import pearsonr
from itertools import combinations
from multiprocessing import Pool, cpu_count
from pyitlib import discrete_random_variable as drv

bin_widths = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]) # various bin widths for testing measurement values.

def getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time):
    """
    Get the bins for a given bin width, and latest spike time.
    Arguments:  spike_time_dict, dict, cell_id => spike_times
                bin_width, float,
                spon_start_time, the time at which spontaneous behaviour begins
    Returns:    numpy.array, the bin boundaries
    """
    max_spike_time = np.array([v.max() for v in spike_time_dict.values()]).max()
    num_bins = np.ceil((max_spike_time - spon_start_time)/bin_width).astype(int)
    end_time = spon_start_time + num_bins * bin_width
    return np.arange(spon_start_time, end_time + bin_width, bin_width)

def getCellSpikeCountBinFrame(cell_id, spike_time_dict, spike_count_bins):
    """
    Helper function for getSpikeCountBinFrame.
    """
    num_rows = spike_count_bins.size - 1
    spike_counts = np.histogram(spike_time_dict[cell_id], bins=spike_count_bins)[0]
    bin_starts = spike_count_bins[:-1]
    bin_stops = spike_count_bins[1:]
    return pd.DataFrame({'cell_id':np.repeat(cell_id, num_rows), 'spike_count':spike_counts, 'bin_start_time':bin_starts, 'bin_stop_time':bin_stops})

def getSpikeCountBinFrame(cell_ids, spike_time_dict, spike_count_bins):
    """
    For getting a frame showing the spike counts with bin times.
    Arguments:  cell_ids, numpy.array (int), the cell ids
                spike_time_dict, dict, cell_id => spike_times
                spike_count_bins, numpy.array (float), the bin boundaries
    Returns:    DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time

    DEPRECATED
    """
    cell_ids = np.array([cell_ids]) if np.isscalar(cell_ids) else cell_ids
    return pd.concat([getCellSpikeCountBinFrame(cell_id, spike_time_dict, spike_count_bins) for cell_id in cell_ids], ignore_index=True)

def getSpikeCountCorrelationsForPair(pair, spike_count_frame):
    """
    For calculating the Pearson correlation coefficient and the shuffled correlation between the spike counts of the cells in 'pair'.
    Arguments:  pair, numpy.array (int, 2), cell ids of the pair
                spike_count_frame, DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time
    Returns:    corr, float, Pearson correlation coefficient of the spike counts
                corr_pv, float, p-value of 'corr', the probability that two random samples would have a correlation equal to 'corr'
                shuff_corr, float, Pearson correlation coefficient between the shuffled spike counts of the first cell, and the spike counts of the second cell
                shuff_corr_pv, float, the p-value of shuff_corr
    """
    first_spike_counts = spike_count_frame.loc[spike_count_frame.cell_id == pair[0], 'spike_count'].values
    second_spike_counts = spike_count_frame.loc[spike_count_frame.cell_id == pair[1], 'spike_count'].values
    corr, corr_pv = pearsonr(first_spike_counts, second_spike_counts)
    np.random.shuffle(first_spike_counts)
    shuff_corr, shuff_corr_pv = pearsonr(first_spike_counts, second_spike_counts)
    return corr, corr_pv, shuff_corr, shuff_corr_pv

def getMutualInfoForPair(pair, spike_count_frame):
    """
    For calculating the mutual information and suffled mutual information between two spike counts.
    Arguments:  pair, numpy.array (int, 2), cell ids of the pair
                spike_count_frame, DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time
    Returns:    plugin_mi, float, plugin mutual information
                plugin_shuff_mi, float, plugin shuffled mutual information
    """
    first_spike_counts = spike_count_frame.loc[spike_count_frame.cell_id == pair[0], 'spike_count'].values
    second_spike_counts = spike_count_frame.loc[spike_count_frame.cell_id == pair[1], 'spike_count'].values
    first_response_alphabet = np.arange(first_spike_counts.max()+1)
    second_response_alphabet = np.arange(second_spike_counts.max()+1)
    plugin_mi = np.max([0, drv.information_mutual(X=first_spike_counts, Y=second_spike_counts, Alphabet_X=first_response_alphabet, Alphabet_Y=second_response_alphabet)])
    np.random.shuffle(second_spike_counts)
    plugin_shuff_mi = np.max([0, drv.information_mutual(X=first_spike_counts, Y=second_spike_counts, Alphabet_X=first_response_alphabet, Alphabet_Y=second_response_alphabet)])
    return plugin_mi, plugin_shuff_mi

def getAnalysisFrameForCells(cell_ids, spike_count_frame):
    """
    For measuring the correlation coefficients and mutual information between the cells, using the counts in the spike_count_frame.
    Arguments:  cell_ids, numpy.array (int), the cell ids
                spike_count_frame, DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time
    Returns:    analysis_frame, DataFrame, corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv
    """
    pairs = np.array(list(combinations(cell_ids, 2)))
    num_pairs = pairs.shape[0]
    corr_coef = np.zeros(num_pairs)
    corr_pv = np.zeros(num_pairs)
    shuff_corr = np.zeros(num_pairs)
    shuff_corr_pv = np.zeros(num_pairs)
    plugin_mi = np.zeros(num_pairs)
    plugin_shuff_mi = np.zeros(num_pairs) # parallelise this.
    for i,pair in enumerate(pairs):
        corr_coef[i], corr_pv[i], shuff_corr[i], shuff_corr_pv[i] = getSpikeCountCorrelationsForPair(pair, spike_count_frame)
        plugin_mi[i], plugin_shuff_mi[i], qe_mi[i], qe_shuff_mi[i] = getMutualInfoForPair(pair, spike_count_frame)
    analysis_frame = pd.DataFrame({'first_cell_id':pairs[:,0], 'second_cell_id':pairs[:,1], 'corr_coef':corr_coef, 'corr_pv':corr_pv, 'shuff_corr':shuff_corr, 'shuff_corr_pv':shuff_corr_pv, 'plugin_mi':plugin_mi, 'plugin_shuff_mi':plugin_shuff_mi})
    return analysis_frame

def getAnalysisFrameForCellsParallel(cell_ids, spike_count_frame):
    """
    For measuring the correlation coefficients and mutual information between the cells, using the counts in the spike_count_frame.
    Arguments:  cell_ids, numpy.array (int), the cell ids
                spike_count_frame, DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time
    Returns:    analysis_frame, DataFrame, corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv
    """
    pairs = np.array(list(combinations(cell_ids, 2)))
    num_pairs = pairs.shape[0]
    num_workers = cpu_count()
    chunk_size = np.ceil(num_pairs/num_workers).astype(int)
    with Pool(num_workers) as pool:
        corr_future = pool.starmap_async(getSpikeCountCorrelationsForPair, zip(pairs, [spike_count_frame] * num_pairs))
        info_future = pool.starmap_async(getMutualInfoForPair, zip(pairs, [spike_count_frame] * num_pairs))
        corr_future.wait()
        info_future.wait()


def getAllBinsFrameForCells(cell_ids, spike_time_dict, spon_start_time):
    """
    For getting a big analysis frame for all bin widths, with a bin width column.
    Arguments:  cell_ids, numpy.array (int), the cell ids
                spike_time_dict, dict, cell_id => spike_times
                spon_start_time, the time at which spontaneous behaviour begins
    Returns:    DataFrame, corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv, bin_width
    """
    analysis_frames = []
    for bin_width in enumerate(bin_widths):
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
        spike_count_bins = getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
        spike_count_frame = getSpikeCountBinFrame(cell_ids, spike_time_dict, spike_count_bins)
        analysis_frame = getAnalysisFrameForCells(cell_ids, spike_count_frame)
        analysis_frame['bin_width'] = bin_width
        analysis_frames += [analysis_frame]
    return pd.concat(analysis_frames, ignore_index=True)
