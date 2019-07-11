import os, sys, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
if float(sys.version[:3])<3.0:
    from pyentropy import DiscreteSystem

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

def getSpikeCountBinFrame(cell_ids, spike_time_dict, spike_count_bins):
    """
    For getting a frame showing the spike counts with bin times.
    Arguments:  cell_ids, numpy.array (int), the cell ids
                spike_time_dict, dict, cell_id => spike_times
                spike_count_bins, numpy.array (float), the bin boundaries
    Returns:    DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time
    """
    def getCellSpikeCountBinFrame(cell_id, spike_time_dict, spike_count_bins):
        num_rows = spike_count_bins.size - 1
        spike_counts = np.histogram(spike_time_dict[cell_id], bins=spike_count_bins)[0]
        bin_starts = spike_count_bins[:-1]
        bin_stops = spike_count_bins[1:]
        return pd.DataFrame({'cell_id':np.repeat(cell_id, num_rows), 'spike_count':spike_counts, 'bin_start_time':bin_starts, 'bin_stop_time':bin_stops})
    
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
                qe_mi, float, quadratic extrapolation corrected mutual information
                qe_shuff_mi, quadratic extrapolation corrected mutual information
    """
    first_spike_counts = spike_count_frame.loc[spike_count_frame.cell_id == pair[0], 'spike_count']
    second_spike_counts = spike_count_frame.loc[spike_count_frame.cell_id == pair[1], 'spike_count']
    first_response_dims = [1, 1 + first_spike_counts.max()]
    second_response_dims = [1, 1 + second_spike_counts.max()]
    discrete_system = DiscreteSystem(first_spike_counts, first_response_dims, second_spike_counts, second_response_dims)
    discrete_system.calculate_entropies(method='plugin', calc=['HX', 'HXY', 'HiXY', 'HshXY'])
    plugin_mi = np.max([0, discrete_system.I()])
    plugin_shuff_mi = np.max([discrete_system.Ish()])
    discrete_system.calculate_entropies(method='qe', calc=['HX', 'HXY'])
    discrete_system.calculate_entropies(method='qe', calc=['HX', 'HXY', 'HiXY', 'HshXY'])
    qe_mi = np.max([0, discrete_system.I()])
    qe_shuff_mi = np.max([discrete_system.Ish()])
    return plugin_mi, plugin_shuff_mi, qe_mi, qe_shuff_mi

