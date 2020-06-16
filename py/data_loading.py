import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from multiprocessing import Pool, cpu_count

mouse_names = np.array(['Krebs', 'Waksman', 'Robbins'])
spon_start_times = np.array([3811, 3633, 3323])
num_probes = 8
probe_ids = np.arange(num_probes)

regions = np.array(['FrCtx','FrMoCtx','SomMoCtx','SSCtx','V1','V2','RSP','CP','LS','LH','HPF','TH','SC','MB'])

def loadSpikesForMouse(mouse_name, mat_dir):
    """
    For loading the spike file for the given mouse name.
    Arguments:  mouse_name, string, the name of the mouse
                mat_dir, the directory where the file can be found
    Returns:    the spike data
    """
    spikes_file_name = 'spks' + mouse_name + '_Feb18.mat'
    return loadmat(os.path.join(mat_dir, spikes_file_name))['spks'][0]

def loadVideoDataForMouse(mouse_name, mat_dir):
    """
    For loading the video file for the given mouse name.
    Arguments:  mouse_name, string, the name of the mouse
                mat_dir, the directory where the file can be found
    Returns:    the video data
    """
    face_file_name = mouse_name + '_face_proc.mat'
    return loadmat(os.path.join(mat_dir, face_file_name))

def getSpikesForCell(cell_id, cell_info, mouse_spikes, spon_start_time):
    """
    Helper function for all of the SpikeTimeDict function.
    """
    specific_cell_info = cell_info.loc[cell_id]
    probe_spikes = mouse_spikes[specific_cell_info['probe_id']]
    cell_spike_times = probe_spikes['st'].flatten()[probe_spikes['clu'].flatten() == specific_cell_info['mouse_probe_cell_id']]
    return cell_spike_times[cell_spike_times >= spon_start_time]

def loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir):
    """
    For loading a dictionary of cell_id => spike times.
    Arguments:  mouse_name, string, the name of the mouse
                cell_ids, the cell ids for which we want the spike times
                cell_info, pandas.DataFrame, a table of information on the cells, gives the probe id, and the mouse_probe_cell_id
                mat_dir, the directory where the spikes file can be found
    Returns:    spike_dict, dictionary, cell_id => spike times
    """
    cell_ids = np.array([cell_ids]) if np.isscalar(cell_ids) else cell_ids
    spon_start_time = spon_start_times[np.flatnonzero(mouse_names == mouse_name)[0]]
    mouse_spikes = loadSpikesForMouse(mouse_name, mat_dir)
    relevant_cell_info = cell_info.loc[cell_ids]
    required_probes = relevant_cell_info.probe_id.unique()
    spike_times_for_cells = [getSpikesForCell(cell_id, cell_info, mouse_spikes, spon_start_time) for cell_id in cell_ids]
    return {cell_id:spike_times for (cell_id, spike_times) in zip(cell_ids, spike_times_for_cells)}

def loadSpikeTimeDictInefficient(mouse_name, cell_ids, cell_info, mat_dir):
    """
    For loading a dictionary of cell_id => spike times.
    Arguments:  mouse_name, string, the name of the mouse
                cell_ids, the cell ids for which we want the spike times
                cell_info, pandas.DataFrame, a table of information on the cells, gives the probe id, and the mouse_probe_cell_id
                mat_dir, the directory where the spikes file can be found
    Returns:    spike_dict, dictionary, cell_id => spike times

    DEPRECATED: This parallelised version is slower than the single-threaded version.
    """
    cell_ids = np.array([cell_ids]) if np.isscalar(cell_ids) else cell_ids
    spon_start_time = spon_start_times[np.flatnonzero(mouse_names == mouse_name)[0]]
    mouse_spikes = loadSpikesForMouse(mouse_name, mat_dir)
    relevant_cell_info = cell_info.loc[cell_ids]
    required_probes = relevant_cell_info.probe_id.unique()
    num_workers = cpu_count()
    chunk_size = np.max([1, np.floor(cell_ids.size/num_workers).astype(int)])
    with Pool(num_workers) as pool:
        spike_times_for_cells_futures = pool.starmap_async(getSpikesForCell, zip(cell_ids, [cell_info] * cell_ids.size, [mouse_spikes] * cell_ids.size, [spon_start_time] * cell_ids.size), chunksize=chunk_size)
        spike_times_for_cells_futures.wait()
    return {cell_id:spike_times for (cell_id, spike_times) in zip(cell_ids, spike_times_for_cells_futures.get())}

def loadSpikeTimeDictScoop(mouse_name, cell_ids, cell_info, mat_dir): # good idea, but requires a rewrite
    """
    For loading a dictionary of cell_id => spike times.
    Arguments:  mouse_name, string, the name of the mouse
                cell_ids, the cell ids for which we want the spike times
                cell_info, pandas.DataFrame, a table of information on the cells, gives the probe id, and the mouse_probe_cell_id
                mat_dir, the directory where the spikes file can be found
    Returns:    spike_dict, dictionary, cell_id => spike times
    """
    cell_ids = np.array([cell_ids]) if np.isscalar(cell_ids) else cell_ids
    spon_start_time = spon_start_times[np.flatnonzero(mouse_names == mouse_name)[0]]
    mouse_spikes = loadSpikesForMouse(mouse_name, mat_dir)
    relevant_cell_info = cell_info.loc[cell_ids]
    required_probes = relevant_cell_info.probe_id.unique()
    spike_times_for_cells = futures.mapReduce(getSpikesForCell, np.hstack, zip(cell_ids, [cell_info] * cell_ids.size, [mouse_spikes] * cell_ids.size, [spon_start_time] * cell_ids.size))
    return {cell_id:spike_times for (cell_id, spike_times) in zip(cell_ids, spike_times_for_cells)}

def loadSpikeCountFrame(mouse_name, bin_width, npy_dir):
    """
    For loading one of the spike count frames from file.
    Arguments:  mouse_name, string, the name of the mouse
                bin_width, float, the bin width
    Returns:    DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time, bin_width
    """
    file_name = os.path.join(npy_dir, 'spike_count_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'spike_counts.npy')
    return pd.read_pickle(file_name)

def loadAnalysisFrame(mouse_name, bin_width, file_dir):
    """
    For loading one of the analysis frames from file.
    Arguments:  mouse_name, string, the name of the mouse.
                bin_width, float, the bin width
                file_dir, string, either npy_dir, or csv_dir
    Returns:    analysis_frame, DataFrame, corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv
    """
    file_ext = os.path.basename(file_dir)
    file_name = os.path.join(file_dir, 'analysis_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'analysis.' + file_ext)
    analysis_frame = pd.read_pickle(file_name) if file_ext == 'npy' else pd.read_csv(file_name)
    return analysis_frame

def loadActiveCellFrame(mouse_name, bin_width, npy_dir):
    """
    For loading one of the active cell frames from file.
    Arguments:  mouse_name, string, the name of the mouse.
                bin_width, float, the bin width
    Returns:    active_cell_frame, DataFrame, bin_start_time, bin_end_time, num_active_cells, bin_width, mouse_name, region
    """
    file_name = os.path.join(npy_dir, 'active_cell_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_active.npy')
    return pd.read_pickle(file_name)

def loadFiringRateFrame(mouse_name, bin_width, npy_dir):
    """
    For loading one of the firing rate frames from file.
    Arguments:  mouse_name, string, the name of the mouse.
                bin_width, float, the bin width
    Returns:    firing_rate_frame, DataFrame, cell_id, spike_count_mean, spike_count_std, firing_rate, firing_std
    """
    file_name = os.path.join(npy_dir, 'firing_rate_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_firing.npy')
    return pd.read_pickle(file_name)

def loadMeasureStatFile(bin_width, csv_dir):
    """
    For loading the csv containing mean and std values for correlation, shuffled correlation, and info across and within regions.
    Arguments:  bin_width, float, the bin width to use.
                csv_dir, string, directory
    Returns:    pandas DataFrame
    """
    return pd.read_csv(os.path.join(csv_dir, 'measure_statistics', 'measure_statistics_' + str(bin_width).replace('.', 'p') + '.csv'), index_col=0)

def loadCommunityInfo(mouse_name, bin_width, npy_dir, correction='rectified', correlation_type='total', is_signal=True):
    """
    For loading in a cell_info frame with detected communities attached.
    Arguments:  mouse_name, string
                bin_width, float,
                npy_dir, string, directory
                correction, rectified or absolute
                is_signal, flag for signal or noise
    Returns:    pandas DataFrame
    """
    sig_or_noise = '_signal' if is_signal else '_noise'
    file_base_name = mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + correction + '_' + correlation_type  + sig_or_noise + '_final_cell_info.pkl'
    return pd.read_pickle(os.path.join(npy_dir, 'communities', file_base_name))

def loadConditionalAnalysisFrame(mouse_name, bin_width, csv_dir):
    """
    For loading the analysis frame with the conditional correlations for each pair.
    Arguments:  mouse_name, string,
                bin_width,  float,
                csv_dir,    string,
    Returns:    pandas DataFrame
    """
    file_name = os.path.join(csv_dir, 'conditional_analysis_frames', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + 'conditional_analysis.csv')
    cond_analysis_frame = pd.read_csv(file_name)
    cond_analysis_frame['bin_width'] = bin_width
    return cond_analysis_frame

def loadLinearModelsFrame(mouse_name, bin_width, csv_dir):
    """
    For loading the analysis frame with the conditional correlations for each pair.
    Arguments:  mouse_name, string,
                bin_width,  float,
                csv_dir,    string,
    Returns:    pandas DataFrame
    """
    file_name = os.path.join(csv_dir, 'linear_model_frames', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + 'linear_models.csv')
    return pd.read_csv(file_name)

def loadExpCondCovMatrix(mouse_name, bin_width, npy_dir):
    """
    For loading the matrix of expected conditional covariance.
    Arguments:  mouse_name, str
                bin_width, float,
                npy_dir, str,
    Returns:    numpy array
    """
    return np.load(os.path.join(npy_dir, 'exp_cond_cov', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + 'exp_cond_cov.npy'))

def loadCovCondExpMatrix(mouse_name, bin_width, npy_dir):
    """
    For loading the matrix of covariances of conditional expectations.
    Arguments:  mouse_name, str
                bin_width, float,
                npy_dir, str,
    Returns:    numpy array
    """
    return np.load(os.path.join(npy_dir, 'cov_cond_exp', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + 'cov_cond_exp.npy'))
