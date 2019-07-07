import os, sys
import numpy as np
import pandas as pd
import datetime as dt
from scipy.io import loadmat
from itertools import product

proj_dir = os.path.join(os.environ['PROJ'], 'Eight_Probe')
mat_dir = os.path.join(proj_dir, 'mat')
csv_dir = os.path.join(proj_dir, 'csv')

mouse_names = np.array(['Krebs', 'Waksman', 'Robbins'])
spon_start_times = np.array([3811, 3633, 3323])
probe_borders = loadmat(os.path.join(mat_dir, 'probeBorders.mat'))
num_probes = 8
probe_ids = np.arange(num_probes)

regions = np.array(['FrCtx','FrMoCtx','SomMoCtx','SSCtx','V1','V2','RSP','CP','LS','LH','HPF','TH','SC','MB'])

def loadSpikesForMouse(mouse_name):
    spikes_file_name = 'spks' + mouse_name + '_Feb18.mat'
    return loadmat(os.path.join(mat_dir, spikes_file_name))['spks'][0]

def getMouseProbeFrame(mouse_name, probe_id, probe_borders):
    mouse_spikes = loadSpikesForMouse(mouse_name)
    mouse_probe_cell_ids, total_spike_counts = np.unique(mouse_spikes[probe_id]['clu'], return_counts=True)
    mouse_probe_frame = pd.DataFrame({'mouse_probe_cell_id':mouse_probe_cell_ids, 'mouse_name':mouse_name.repeat(mouse_probe_cell_ids.size), 
        'probe_id':np.repeat(probe_id, mouse_probe_cell_ids.size), 'height':mouse_spikes['Wheights'][probe_id].flatten(), 
        'total_spikes':total_spike_counts, 'cell_region':np.repeat('', mouse_probe_cell_ids.size)})
    lower_borders = probe_borders['probeBorders'][0][0]['borders'][0][probe_id]['lowerBorder'][:,0]
    upper_borders = probe_borders['probeBorders'][0][0]['borders'][0][probe_id]['upperBorder'][:,0]
    acronyms = probe_borders['probeBorders'][0][0]['borders'][0][probe_id]['acronym'][:,0]
    for i, acronym in enumerate(acronyms):
        lower_border = lower_borders[i][0][0]
        upper_border = upper_borders[i][0][0]
        mouse_probe_frame.loc[(lower_border <= mouse_probe_frame.height) & (mouse_probe_frame.height < upper_border), 'cell_region'] = acronym[0]
    return mouse_probe_frame

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading...')
cell_info = pd.concat([getMouseProbeFrame(mouse_name, probe_id, probe_borders) for mouse_name, probe_id in product(mouse_names, probe_ids)], ignore_index=True)
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving...')
cell_info.to_csv(os.path.join(csv_dir, 'cell_info.csv'))
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

