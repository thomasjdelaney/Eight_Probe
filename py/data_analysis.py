import os, sys, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import pearsonr
from itertools import combinations, product
from multiprocessing import Pool, cpu_count
from pyitlib import discrete_random_variable as drv
from functools import reduce
from sklearn.preprocessing import normalize

bin_widths = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]) # various bin widths for testing measurement values.
augmented_bin_widths = np.concatenate([[0.001, 0.002, 0.005], bin_widths])
selected_bin_widths = np.array([0.005, 0.01, 0.05, 0.1, 0.2, 1.0, 2.0, 3.0])


def getRegionallyDistributedCells(cell_info, num_cells):
    """
    For getting num_cells cell_ids distributed across the different available brain regions. cell_info should be prefiltered here.
    Will return the same number of cells for each region, so the number of cells returned may be less than the number requested!!!
    Arguments:  cell_info, DataFrame, *This should be filtered before using this function*
                num_cells, the number of cells to include
    Returns:    distributed_cells, numpy.array int, cell_ids
    """
    regions = cell_info.cell_region.unique()
    num_regions = regions.size
    cells_per_region = num_cells // num_regions
    distributed_cells = np.zeros(cells_per_region * num_regions, dtype=int)
    cells_added = 0
    for region in regions:
        region_cell_info = cell_info.loc[cell_info.cell_region == region]
        num_region_cells = region_cell_info.shape[0]
        if cells_per_region > num_region_cells:
            cells_to_add = region_cell_info.index.values
        else:
            cells_to_add = np.random.choice(cell_info.loc[cell_info.cell_region == region].index.values, cells_per_region, replace=False)
        distributed_cells[np.arange(cells_added, cells_added + cells_to_add.size)] = cells_to_add
        cells_added += cells_to_add.size
    more_to_add = distributed_cells.size - cells_added
    if more_to_add > 0:
        cells_so_far = np.setdiff1d(distributed_cells, 0)
        all_cells = cell_info.index.values
        cells_to_add = np.random.choice(np.setdiff1d(all_cells, cells_so_far), more_to_add, replace=False)
        distributed_cells[np.arange(cells_added, cells_added + cells_to_add.size)] = cells_to_add
    return distributed_cells

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

def getRelevantMotionSVD(mouse_face, time_bins):
    """
    Get the relevant motion SVD values from the given mouse face dict.
    Arguments:  mouse_face, dict, contains all the information from the mouse videos
                time_bins, numpy.array (float),  the time bin boundaries for binning spikes
    Returns:    numpy.array (float), relevant mouse face times
                numpy.array (float), 2-d (num time bins, num components), trimmed mouse_face['motionSVD']
    """
    mouse_face_times = mouse_face.get('times')[0]
    is_relevant = (time_bins[0] <= mouse_face_times) & (mouse_face_times <= time_bins[-1])
    return mouse_face_times[is_relevant], mouse_face.get('motionSVD')[is_relevant, :]

def getSpikeCountsFromTimes(spike_times, spike_count_bins):
    return np.histogram(spike_times, bins=spike_count_bins)[0]

def getSpikeCountDict(spike_time_dict, bin_width, spon_start_time):
    """
    For getting a dictionary cell_id => spike counts.
    Arguments:  spike_time_dict, cell_id => spike times
                bin_width,
                spon_start_time, the time at which spontaneous behaviour begins
    Returns:    dictionary cell_id => spike counts
    """
    spike_count_bins = getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
    num_cells = len(spike_time_dict)
    with Pool() as pool:
        spike_count_futures = pool.starmap_async(getSpikeCountsFromTimes, zip(list(spike_time_dict.values()), [spike_count_bins] * num_cells))
        spike_count_futures.wait()
    return dict(zip(list(spike_time_dict.keys()), spike_count_futures.get()))

def getSpikeCountFromTimesAndBins(cell_id, spike_time_dict, spike_count_bins):
    """
    helper function for getActiveCellBinFrame. Gets binned spike counts.
    """
    return np.histogram(spike_time_dict[cell_id], bins=spike_count_bins)[0]

def getCellSpikeCountBinFrame(cell_id, spike_time_dict, spike_count_bins):
    """
    Helper function for getSpikeCountBinFrame.
    """
    num_rows = spike_count_bins.size - 1
    spike_counts = np.histogram(spike_time_dict[cell_id], bins=spike_count_bins)[0]
    bin_starts = spike_count_bins[:-1]
    bin_stops = spike_count_bins[1:]
    return pd.DataFrame({'cell_id':np.repeat(cell_id, num_rows), 'spike_count':spike_counts, 'bin_start_time':bin_starts, 'bin_stop_time':bin_stops})

def getActiveCellBinFrame(cell_ids, spike_time_dict, spike_count_bins):
    """
    For getting a frame showing the number of active cells in each bin.
    Arguments:  cell_ids, numpy.array (int), the cell ids
                spike_time_dict, dict, cell_id => spike_times
                spike_count_bins, numpy.array (float), the bin boundaries
    Returns:    DataFrame, bin_start_time, bin_stop_time, num_active_cells
    """
    active_cell_frame = pd.DataFrame({'bin_start_time':spike_count_bins[:-1], 'bin_stop_time':spike_count_bins[1:], 'num_active_cells':0})
    with Pool() as pool:
        spike_counts_future = pool.starmap_async(getSpikeCountFromTimesAndBins, zip(cell_ids, [spike_time_dict]*cell_ids.size, [spike_count_bins]*cell_ids.size))
        spike_counts_future.wait()
    total_active_cells = reduce(lambda x, y: x + (y>0), spike_counts_future.get(), np.zeros(active_cell_frame.shape[0], dtype=int))
    active_cell_frame.loc[:, 'num_active_cells'] = total_active_cells
    return active_cell_frame

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

def getSpikeCountCorrelationsForPair(pair_count_dict):
    """
    For calculating the Pearson correlation coefficient and the shuffled correlation between the spike counts of the cells in 'pair'.
    Arguments:  pair_count_dict, int => numpy array int, the two keys is the pair, the values are the spike counts
    Returns:    corr, float, Pearson correlation coefficient of the spike counts
                corr_pv, float, p-value of 'corr', the probability that two random samples would have a correlation equal to 'corr'
                shuff_corr, float, Pearson correlation coefficient between the shuffled spike counts of the first cell, and the spike counts of the second cell
                shuff_corr_pv, float, the p-value of shuff_corr
    """
    first_spike_counts, second_spike_counts = np.array(list(pair_count_dict.values()))
    corr, corr_pv = pearsonr(first_spike_counts, second_spike_counts)
    np.random.shuffle(first_spike_counts)
    shuff_corr, shuff_corr_pv = pearsonr(first_spike_counts, second_spike_counts)
    return corr, corr_pv, shuff_corr, shuff_corr_pv

def getPTBiasEstimate(first_spike_counts, second_spike_counts):
    """
    For calculating an estimate of the bias on the measurement of mutual information.
    Arguments:  first_spike_counts, numpy array, spike counts
                second_spike_counts, numpy array, spike counts
    Returns:    bias_estimate, float
    Reference:  Stefano Panzeri, Alessandro Treves, Analytical estimates of limited sampling biases in different information measures, Network: Computation in Neural Systems 7, 87–107, (1996)
    """
    response, stimulus = [second_spike_counts, first_spike_counts] if np.argmax([first_spike_counts.max(), second_spike_counts.max()]) else [first_spike_counts, second_spike_counts] # choosing 'stimulus' as the spike count array with fewer possible responses, we will need to loop through these responses.
    response_bins = np.arange(response.max()+1)
    stimulus_bins = np.unique(stimulus)
    relevant_responses = np.intersect1d(response_bins, response).size
    stimulus_relevant_responses = np.array([np.unique(response[np.flatnonzero(stimulus == s)]).size for s in stimulus_bins])
    brackets = (stimulus_relevant_responses.sum() - stimulus_bins.size) - (relevant_responses - 1)
    coef = -1 / (2 * first_spike_counts.size * np.log(2))
    bias_estimate = coef * brackets if brackets > 0 else 0 # we don't want any positive bias corrections.
    return bias_estimate

def getMutualInfoForPair(pair_count_dict):
    """
    For calculating the mutual information and suffled mutual information between two spike counts.
    Arguments:  pair_count_dict, int => numpy array int, the two keys is the pair, the values are the spike counts
    Returns:    plugin_mi, float, plugin mutual information
                plugin_shuff_mi, float, plugin shuffled mutual information
    """
    first_spike_counts, second_spike_counts = np.array(list(pair_count_dict.values()))
    first_response_alphabet = np.arange(first_spike_counts.max()+1)
    second_response_alphabet = np.arange(second_spike_counts.max()+1)
    plugin_mi = np.max([0, drv.information_mutual(X=first_spike_counts, Y=second_spike_counts, Alphabet_X=first_response_alphabet, Alphabet_Y=second_response_alphabet)])
    bias_estimate = getPTBiasEstimate(first_spike_counts, second_spike_counts)
    bias_corrected_mi = np.max([0, plugin_mi + bias_estimate])
    np.random.shuffle(second_spike_counts)
    plugin_shuff_mi = np.max([0, drv.information_mutual(X=first_spike_counts, Y=second_spike_counts, Alphabet_X=first_response_alphabet, Alphabet_Y=second_response_alphabet)])
    return plugin_mi, plugin_shuff_mi, bias_corrected_mi

def getAnalysisFrameForCellsOld(cell_ids, spike_count_frame):
    """
    For measuring the correlation coefficients and mutual information between the cells, using the counts in the spike_count_frame.
    Arguments:  cell_ids, numpy.array (int), the cell ids
                spike_count_frame, DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time
    Returns:    analysis_frame, DataFrame, corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv

    DEPRECATED: This is not parallelised, and the parallelisation is super effective here.
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
        plugin_mi[i], plugin_shuff_mi[i] = getMutualInfoForPair(pair, spike_count_frame)
    analysis_frame = pd.DataFrame({'first_cell_id':pairs[:,0], 'second_cell_id':pairs[:,1], 'corr_coef':corr_coef, 'corr_pv':corr_pv, 'shuff_corr':shuff_corr, 'shuff_corr_pv':shuff_corr_pv, 'plugin_mi':plugin_mi, 'plugin_shuff_mi':plugin_shuff_mi})
    return analysis_frame

def getAnalysisFrameForCells(cell_ids, spike_count_frame):
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
        corr_future.wait() # map reduce with hstack? Try it.
        info_future.wait()
    corr = np.array(corr_future.get())
    info = np.array(info_future.get())
    analysis_frame = pd.DataFrame({'first_cell_id':pairs[:,0], 'second_cell_id':pairs[:,1], 'corr_coef':corr[:,0], 'corr_pv':corr[:,1], 'shuff_corr':corr[:,2], 'shuff_corr_pv':corr[:,3], 'plugin_mi':info[:,0], 'plugin_shuff_mi':info[:,1]})
    return analysis_frame

def getAllBinsFrameForCells(cell_ids, spike_time_dict, spon_start_time):
    """
    For getting a big analysis frame for all bin widths, with a bin width column.
    Arguments:  cell_ids, numpy.array (int), the cell ids
                spike_time_dict, dict, cell_id => spike_times
                spon_start_time, the time at which spontaneous behaviour begins
    Returns:    DataFrame, corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv, bin_width

    NB A subset of the cells must be used here, otherwise memory issues will occur.
    """
    analysis_frames = []
    for bin_width in bin_widths:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
        spike_count_bins = getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
        spike_count_frame = getSpikeCountBinFrame(cell_ids, spike_time_dict, spike_count_bins)
        analysis_frame = getAnalysisFrameForCells(cell_ids, spike_count_frame)
        analysis_frame['bin_width'] = bin_width
        analysis_frames += [analysis_frame]
    return pd.concat(analysis_frames, ignore_index=True)

def getFiringRateFrameFromSpikeCountFrame(spike_count_frame, bin_width):
    """
    For making a firing rate frame given a spike count frame, and the bin width.
    Arguments:  spike_count_frame, DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time, bin_width
    Returns:    firing_rate_frame, DataFrame, cell_id, spike_count_mean, spike_count_std, firing_rate, firing_std
    """
    column_names = ['cell_id', 'spike_count_mean', 'spike_count_std', 'firing_rate', 'firing_std']
    firing_rate_frame = spike_count_frame[['cell_id', 'spike_count']].groupby('cell_id').agg(['mean', 'std'])
    firing_rate_frame = firing_rate_frame.reset_index()
    firing_rate_frame.columns = column_names[:3]
    firing_rate_frame['firing_rate'] = firing_rate_frame['spike_count_mean']/bin_width
    firing_rate_frame['firing_std'] = firing_rate_frame['spike_count_std']/bin_width
    return firing_rate_frame

def getFiringRateDict(cell_id, spike_time_dict, spike_count_bins):
    spike_counts = np.histogram(spike_time_dict[cell_id], bins=spike_count_bins)[0]
    return {'cell_id':cell_id, 'spike_count_mean':spike_counts.mean(), 'spike_count_std':spike_counts.std()}
    
def reduceFiringRateDicts(first_dict, second_dict):
    """
    A function for use with map-reduce. Takes two dictionaries with the same keys and concatenates their values.
    """
    return {k:np.append(first_dict[k], second_dict[k]) for k in first_dict.keys()}

def getFiringRateFrameFromSpikeTimeDict(spike_time_dict, bin_width, spon_start_time):
    """
    For making a firing rate frame given a spike_time_dict, and the bin width.
    Arguments:  spike_time_dict, dict, cell_id => spike_times
                bin_width, float
    Returns:    firing_rate_frame, DataFrame, cell_id, spike_count_mean, spike_count_std, firing_rate, firing_std
    """
    num_cells = len(spike_time_dict)
    init_dict = {'cell_id':np.array([], dtype=int), 'spike_count_mean':np.array([]), 'spike_count_std':np.array([])}
    spike_count_bins = getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
    with Pool() as pool:
        dict_future = pool.starmap_async(getFiringRateDict, zip(list(spike_time_dict.keys()), [spike_time_dict] * num_cells, [spike_count_bins] * num_cells))
        dict_future.wait()
    firing_rate_frame = pd.DataFrame.from_dict(reduce(reduceFiringRateDicts, dict_future.get(), init_dict))
    firing_rate_frame['firing_rate'] = firing_rate_frame['spike_count_mean']/bin_width
    firing_rate_frame['firing_std'] = firing_rate_frame['spike_count_std']/bin_width
    firing_rate_frame.sort_values('cell_id', inplace=True)
    firing_rate_frame.reset_index(drop=True, inplace=True)
    return firing_rate_frame

def getFiringRateFrameFromSpikeCountDict(spike_count_dict, bin_width):
    """
    For making a firing rate frame given a spike_count_dict, and the bin width.
    Arguments:  spike_count_dict, dict, cell_id => spike_counts
                bin_width, float,
    Returns:    firing_rate_frame, DataFrame, cell_id, spike_count_mean, spike_count_std, firing_rate, firing_std
    """
    firing_rate_frame = pd.DataFrame.from_dict({'cell_id':np.array(list(spike_count_dict.keys()), dtype=int), 'spike_count_mean':np.array(list(spike_count_dict.values())).mean(axis=1), 'spike_count_std':np.array(list(spike_count_dict.values())).std(axis=1)})
    firing_rate_frame['firing_rate'] = firing_rate_frame['spike_count_mean']/bin_width
    firing_rate_frame['firing_std'] = firing_rate_frame['spike_count_std']/bin_width
    firing_rate_frame.sort_values('cell_id', inplace=True)
    firing_rate_frame.reset_index(drop=True, inplace=True)
    return firing_rate_frame

def getRegionalAnalysisFrame(analysis_frame, region_pair):
    """
    For selecting from the analysis frame according to the regions in region_pair.
    Arguments:  analysis_frame, pandas DataFrame, corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv
                region_pair, double string, a double of strings containing region names
    Return:     filtered analysis frame
    """
    return analysis_frame.loc[((analysis_frame.first_cell_region == region_pair[0]) & (analysis_frame.second_cell_region == region_pair[1])) | ((analysis_frame.first_cell_region == region_pair[1]) & (analysis_frame.second_cell_region == region_pair[0]))]

def joinCellAnalysis(cell_info, analysis_frame):
    """
    For joining cell_info on an analysis_frame giving the regions of the first and second cells.
    Arguments:  cell_info, the cell info table
                analysis_frame, pandas DataFrame, corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv
    Returns:    analysis frame with extra columns.
    """
    analysis_frame = analysis_frame.join(cell_info['cell_region'], how='left', on='first_cell_id')
    analysis_frame = analysis_frame.rename(columns={'cell_region':'first_cell_region'})
    analysis_frame = analysis_frame.join(cell_info['cell_region'], how='left', on='second_cell_id')
    analysis_frame = analysis_frame.rename(columns={'cell_region':'second_cell_region'})
    return analysis_frame

def getRegionalMeasureAggFrame(measure_agg_frame, region_pair):
    """
    For selecting from the measure aggregation frame according to the given region pair
    Arguments:  measure_agg_frame, pandas DataFrame, contains all measurements aggregated over and between regions
                region_pair, double string, a double of strings containing region names
    Returns:    filtered measure_agg_frame
    """
    return measure_agg_frame.loc[((measure_agg_frame.first_region == region_pair[0]) & (measure_agg_frame.second_region == region_pair[1])) | ((measure_agg_frame.first_region == region_pair[1]) & (measure_agg_frame.second_region == region_pair[0]))]

def getRegionalMeasureMatrix(measure_frame, measure, mouse_name=None, regions=np.array(None)):
    """
    For getting a symmetric martix of pairwise measurements. mouse_name and region pairs are optional.
    Arguments:  measure_frame, measure_agg_frame, pandas DataFrame, contains all measurements aggregated over and between regions
                measure, one of 'mean_corr', 'mean_shuff_corr', 'mean_info', 'std_corr', 'std_shuff_corr', 'std_info'
    Returns:    a matrix of measurement values, the size of the number of regions
    """
    measure_frame = measure_frame.loc[measure_frame.mouse_name == mouse_name] if mouse_name != None else measure_frame
    if np.all(regions == None):
        regions = measure_frame[(measure_frame.first_region == measure_frame.second_region)].sort_values(measure, ascending=False).first_region.unique()
        region_pairs = product(regions, regions)
    else: 
        region_pairs = product(regions, regions)
    measure_matrix = np.zeros([regions.size, regions.size])
    for region_pair in region_pairs:
        first_ind = np.flatnonzero(regions == region_pair[0])[0]
        second_ind = np.flatnonzero(regions == region_pair[1])[0]
        measure_value = getRegionalMeasureAggFrame(measure_frame, region_pair).iloc[0][measure]
        measure_matrix[first_ind, second_ind] = measure_value
        measure_matrix[second_ind, first_ind] = measure_value
    return measure_matrix, regions

def getSpikeCountHistsForMotionSVD(mouse_face, spike_count_dict, time_bins, num_bins_svd=50):
    """
    For getting the histogram of spike counts by SVD comp value.
    mouse_face = ep.loadVideoDataForMouse(mouse_name, mat_dir)
    mouse_face = ep.getSpikeCountHistsForMotionSVD(mouse_face, spike_count_dict, ep.getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time))
    Arguments:  mouse_face, dict, contains all the information from the motion videos.
                spike_count_dict, dictionary of arrays of spike counts.
                time_bins, numpy.array (float), the bins for the whole experiment
                num_bins_svd, int, the number of bins to split the SVD components into.
    Returns:    mouse_face, dict, adds keys svd_times, svd_comps, spike_count_array, svd_spike_count_hists, svd_hists, svd_bin_borders
    """
    total_exp_time = time_bins[-1] - time_bins[0]
    svd_times, svd_comps = getRelevantMotionSVD(mouse_face, time_bins)
    spike_count_array = np.array(list(spike_count_dict.values()))
    total_spike_counts_by_comp = np.empty(shape=(svd_comps.shape[1], len(spike_count_dict), num_bins_svd), dtype=float)
    covariance_matrices_by_comp = np.empty(shape=(svd_comps.shape[1], len(spike_count_dict),len(spike_count_dict), num_bins_svd), dtype=float)
    svd_hists = np.zeros((svd_comps.shape[1], num_bins_svd), dtype=int) # (num comps, num svd bins)
    svd_bin_borders = np.zeros((svd_comps.shape[1], num_bins_svd+1), dtype=float) # (num comps, num svd bins + 1)
    for i,svd_comp in enumerate(svd_comps.T):
        svd_counts, svd_bins = np.histogram(svd_comp, bins=num_bins_svd)
        for j,(svd_bin_start, svd_bin_stop) in enumerate(zip(svd_bins[:-1], svd_bins[1:])):
            svd_bin_value_times = svd_times[np.logical_and(svd_bin_start <= svd_comp, svd_comp < svd_bin_stop)]
            svd_bin_value_time_bin_inds = np.digitize(svd_bin_value_times, time_bins)
            total_spike_counts_by_comp[i, :, j] = spike_count_array[:, svd_bin_value_time_bin_inds-1].sum(axis=1)
            covariance_matrices_by_comp[i, :, :, j] = np.cov(spike_count_array[:, svd_bin_value_time_bin_inds-1])
        svd_hists[i, :] = svd_counts
        svd_bin_borders[i, :] = svd_bins
    total_spike_counts_by_comp[np.isnan(expected_spike_counts_by_comp)] = 0.0
    covariance_matrices_by_comp[np.isnan(covariance_matrices_by_comp)] = 0.0
    svd_dists = normalize(svd_hists, axis=1, norm='l1')
    cond_exp_by_comp = np.divide(np.divide(total_spike_counts_by_comp.swapaxes(0,1), svd_dists), total_exp_time).swapaxes(0,1) # (num comps, num cells, num svd bins) conditional expectations, conditional on svd value.
    # np.outer
    cond_exp_by_comp[np.isnan(cond_exp_by_comp)] = 0.0
    mouse_face['svd_times'] = svd_times
    mouse_face['svd_comps'] = svd_comps
    mouse_face['svd_spike_count_hists'] = svd_spike_count_hists
    mouse_face['svd_dists'] = svd_dists
    mouse_face['svd_bin_borders'] = svd_bin_borders
    mouse_face['svd_cond_exp'] = svd_cond_exp
    return mouse_face

def getMouseFaceCondSpikeCounts(mouse_face, spike_time_dict):
    """
    For reformatting mouse_face['svd_spike_count_hists'] as an array of dictionaries.
    Arguments:  mouse_face, dict, contains all information from the mouse face videos.
                spike_time_dict, dict, cell_id => spike times
    Returns:    mouse_face, with 'svd_spike_count_dicts' key.
    """
    num_comps = mouse_face['svd_comps'].shape[1]
    svd_cond_spike_count_dicts = np.empty(shape=(num_comps,), dtype=dict)
    for i,svd_cond_spike_counts in enumerate(mouse_face['svd_cond_exp']):
        svd_cond_spike_count_dicts[i] = dict(zip(spike_time_dict.keys(), svd_cond_spike_counts))
    mouse_face['svd_cond_spike_count_dicts'] = svd_cond_spike_count_dicts
    return mouse_face
