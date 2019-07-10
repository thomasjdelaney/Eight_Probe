"""
For varying the bin width used from 0.005 to 2 seconds, and taking measurements using these bin widths.
"""
import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product

parser = argparse.ArgumentParser(description='For varying the bin width used from 0.005 to 2 seconds, and taking measurements using these bin widths.')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows',30) # setting display options for terminal display

proj_dir = os.path.join(os.environ['PROJ'], 'Eight_Probe')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images')
mat_dir = os.path.join(proj_dir, 'mat')

sys.path.append(os.environ['PROJ'])
import Eight_Probe.py as ep

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)

def getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time):
    max_spike_time = np.array([v.max() for v in spike_time_dict.values()]).max()
    num_bins = np.ceil((max_spike_time - spon_start_time)/bin_width).astype(int)
    end_time = spon_start_time + num_bins * bin_width
    return np.arange(spon_start_time, end_time + bin_width, bin_width)


mouse_name = ep.mouse_names[0]
spon_start_time = ep.spon_start_times[0]
cell_ids = cell_info[cell_info.mouse_name == mouse_name].index.values
spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
bin_width = 2.0
spike_count_bins = getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
all_spike_counts = np.zeros([spike_count_bins.size-1, cell_ids.size])
for i, cell_id in enumerate(cell_ids):
    all_spike_counts[:,i] = np.histogram(spike_time_dict[cell_id], bins=spike_count_bins)[0]
