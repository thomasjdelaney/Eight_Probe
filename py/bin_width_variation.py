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

def getAnalysisFrameForCells(cell_ids, spike_count_frame):
    pairs = np.array(list(combinations(cell_ids, 2)))
    num_pairs = pairs.shape[0]
    corr_coef = np.zeros(num_pairs)
    corr_pv = np.zeros(num_pairs)
    shuff_corr = np.zeros(num_pairs)
    shuff_corr_pv = np.zeros(num_pairs)
    plugin_mi = np.zeros(num_pairs)
    plugin_shuff_mi = np.zeros(num_pairs)
    qe_mi = np.zeros(num_pairs)
    qe_shuff_mi = np.zeros(num_pairs)
    for i,pair in enumerate(pairs):
        corr_coef[i], corr_pv[i], shuff_corr[i], shuff_corr_pv[i] = ep.getSpikeCountCorrelationsForPair(pair, spike_count_frame)
        plugin_mi[i], plugin_shuff_mi[i], qe_mi[i], qe_shuff_mi[i] = ep.getMutualInfoForPair(pair, spike_count_frame)
    analysis_frame = pd.DataFrame({'first_cell_id':pairs[:,0], 'second_cell_id':pairs[:,1], 'corr_coef':corr_coef, 'corr_pv':corr_pv, 'shuff_corr':shuff_corr, 'shuff_corr_pv':shuff_corr_pv, 'plugin_mi':plugin_mi, 'plugin_shuff_mi':plugin_shuff_mi, 'qe_mi':qe_mi, 'qe_shuff_mi':qe_shuff_mi})
    return analysis_frame

mouse_name = ep.mouse_names[0]
spon_start_time = ep.spon_start_times[0]
cell_ids = cell_info[cell_info.mouse_name == mouse_name].index.values
spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
bin_width = 2.0
spike_count_bins = ep.getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
spike_count_frame = ep.getSpikeCountBinFrame(cell_ids, spike_time_dict, spike_count_bins)


# all_spike_counts = np.zeros([spike_count_bins.size-1, cell_ids.size])
# for i, cell_id in enumerate(cell_ids):
#     all_spike_counts[:,i] = np.histogram(spike_time_dict[cell_id], bins=spike_count_bins)[0]
