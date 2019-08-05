import os
import numpy as np
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
    specific_cell_info = cell_info.loc[cell_id]
    probe_spikes = mouse_spikes[specific_cell_info['probe_id']]
    cell_spike_times = probe_spikes['st'].flatten()[probe_spikes['clu'].flatten() == specific_cell_info['mouse_probe_cell_id']]
    return cell_spike_times[cell_spike_times >= spon_start_time]

def loadSpikeTimeDictOld(mouse_name, cell_ids, cell_info, mat_dir):
    """
    For loading a dictionary of cell_id => spike times.
    Arguments:  mouse_name, string, the name of the mouse
                cell_ids, the cell ids for which we want the spike times
                cell_info, pandas.DataFrame, a table of information on the cells, gives the probe id, and the mouse_probe_cell_id
                mat_dir, the directory where the spikes file can be found
    Returns:    spike_dict, dictionary, cell_id => spike times

    DEPRECATED: New loadSpikeTimeDict is parallelised and is therefore slightly faster, or perhaps much faster.
    """
    cell_ids = np.array([cell_ids]) if np.isscalar(cell_ids) else cell_ids
    spon_start_time = spon_start_times[np.flatnonzero(mouse_names == mouse_name)[0]]
    mouse_spikes = loadSpikesForMouse(mouse_name, mat_dir)
    relevant_cell_info = cell_info.loc[cell_ids]
    required_probes = relevant_cell_info.probe_id.unique()
    spike_times_for_cells = [getSpikesForCell(cell_id, cell_info, mouse_spikes, spon_start_time) for cell_id in cell_ids]
    return {cell_id:spike_times for (cell_id, spike_times) in zip(cell_ids, spike_times_for_cells)}

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
