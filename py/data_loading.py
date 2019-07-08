import os
import numpy as np
from scipy.io import loadmat

mouse_names = np.array(['Krebs', 'Waksman', 'Robbins'])
spon_start_times = np.array([3811, 3633, 3323])
num_probes = 8 
probe_ids = np.arange(num_probes)

regions = np.array(['FrCtx','FrMoCtx','SomMoCtx','SSCtx','V1','V2','RSP','CP','LS','LH','HPF','TH','SC','MB'])

def loadSpikesForMouse(mouse_name, mat_dir):
    spikes_file_name = 'spks' + mouse_name + '_Feb18.mat'
    return loadmat(os.path.join(mat_dir, spikes_file_name))['spks'][0]

def loadVideoDataForMouse(mouse_name, mat_dir):
    face_file_name = mouse_name + '_face_proc.mat'
    return loadmat(os.path.join(mat_dir, face_file_name))

def loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir):
    spike_dict = {}
    spon_start_time = spon_start_times[np.flatnonzero(mouse_names == mouse_name)[0]]
    mouse_spikes = loadSpikesForMouse(mouse_name, mat_dir)
    relevant_cell_info = cell_info.loc[cell_ids] 
    required_probes = relevant_cell_info.probe_id.unique()
    for cell_id in cell_ids:
        specific_cell_info = cell_info.loc[cell_id]
        probe_spikes = mouse_spikes[specific_cell_info['probe_id']]
        cell_spike_times = probe_spikes['st'].flatten()[probe_spikes['clu'].flatten() == specific_cell_info['mouse_probe_cell_id']]
        spike_dict[cell_id] = cell_spike_times[cell_spike_times >= spon_start_time]
    return spike_dict


