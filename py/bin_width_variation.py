"""
For varying the bin width used from 0.005 to 2 seconds, and taking measurements using these bin widths.
"""
import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product, combinations

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

mouse_name = ep.mouse_names[0]
spon_start_time = ep.spon_start_times[0]
cell_ids = cell_info[cell_info.mouse_name == mouse_name].index.values[:10] # using 10 cells for testing
spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
bin_widths = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
a_frames = []
for i,bin_width, in enumerate(bin_widths):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
    spike_count_bins = ep.getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
    spike_count_frame = ep.getSpikeCountBinFrame(cell_ids, spike_time_dict, spike_count_bins)
    analysis_frame = ep.getAnalysisFrameForCells(cell_ids, spike_count_frame)
    a_frames += [analysis_frame]
all_bins = pd.concat(a_frames, ignore_index=True)

# TODO: Need a bin_width column

# all_spike_counts = np.zeros([spike_count_bins.size-1, cell_ids.size])
# for i, cell_id in enumerate(cell_ids):
#     all_spike_counts[:,i] = np.histogram(spike_time_dict[cell_id], bins=spike_count_bins)[0]
