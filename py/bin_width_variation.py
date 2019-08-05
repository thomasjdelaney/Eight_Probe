"""
For varying the bin width used from 0.005 to 2 seconds, and taking measurements using these bin widths.

We save a frame for spike counts for each time bin containing all cells. We also save a frame for each time bin for all pairs.
"""
import os, argparse, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product, combinations
from scoop import futures

parser = argparse.ArgumentParser(description='For varying the bin width used from 0.005 to 4 seconds, and taking measurements using these bin widths.')
parser.add_argument('-s', '--save_file_suffix', help='A suffix for the name of the file in which to save the measurement frame.', type=str, default='_all_bins.npy')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows',30) # setting display options for terminal display

proj_dir = os.path.join(os.environ['PROJ'], 'Eight_Probe')
py_dir = os.path.join(proj_dir, 'py')
npy_dir = os.path.join(proj_dir, 'npy')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images')
mat_dir = os.path.join(proj_dir, 'mat')

sys.path.append(os.environ['PROJ'])
import Eight_Probe.py as ep

def saveSpikeCountFrame(cell_ids, bin_width, spike_time_dict, spon_start_time, mouse_name):
    spike_count_bins = ep.getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
    spike_count_frame = ep.getSpikeCountBinFrame(cell_ids, spike_time_dict, spike_count_bins)
    spike_count_frame['bin_width'] = bin_width
    save_file = os.path.join(npy_dir, 'spike_count_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'spike_counts.npy')
    spike_count_frame.to_pickle(save_file)
    return save_file, spike_count_frame

def getAnalysisFrameForPair(pair, spike_count_frame):
    """
    For getting a dictionary containing measurements for the given pair. This function is most useful for parallel processing
    as there will be a great number of pairs.
    Arguments:  pair, numpy.array (int, 2), cell ids of the pair
                spike_count_frame, spike_count_frame, DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time
    Returns:    Dict,   keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
    """
    corr, corr_pv, shuff_corr, shuff_corr_pv = ep.getSpikeCountCorrelationsForPair(pair, spike_count_frame)
    plugin_mi, plugin_shuff_mi = ep.getMutualInfoForPair(pair, spike_count_frame)
    return pd.DataFrame({'corr_coef': corr, 'corr_pv':corr_pv, 'first_cell_id':pair[0], 'plugin_mi':plugin_mi, 'plugin_shuff_mi':plugin_shuff_mi, 'second_cell_id':pair[1], 'shuff_corr':shuff_corr, 'shuff_corr_pv':shuff_corr_pv}, index=[0])

def reduceAnalysisFrames(first_frame, second_frame):
    return pd.concat([first_frame, second_frame], ignore_index=True)

def saveAnalysisFrame(pairs, spike_count_frame, bin_width, mouse_name):
    analysis_frame = futures.mapReduce(getAnalysisFrameForPair, reduceAnalysisFrames, zip(pairs, [spike_count_frame] * pairs.shape[0]))
    analysis_frame['bin_width'] = bin_width
    save_file = os.path.join(npy_dir, 'analysis_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'analysis.npy')
    analysis_frame.to_pickle(save_file)
    return save_file

# def main():
if __name__ == "__main__":
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for m,mouse_name in enumerate(ep.mouse_names):
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        spon_start_time = ep.spon_start_times[m]
        cell_ids = cell_info[cell_info.mouse_name == mouse_name].index.values[:30] # using 10 cells for testing
        spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
        pairs = np.array(list(combinations(cell_ids, 2)))
        for bin_width in ep.bin_widths:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
            spike_count_save_file, spike_count_frame = saveSpikeCountFrame(cell_ids, bin_width, spike_time_dict, spon_start_time, mouse_name)
            analysis_frame = futures.mapReduce(getAnalysisFrameForPair, reduceAnalysisFrames, list(zip(pairs, [spike_count_frame] * pairs.shape[0])))
            analysis_frame['bin_width'] = bin_width
            save_file = os.path.join(npy_dir, 'analysis_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'analysis.npy')
            analysis_frame.to_pickle(save_file)
            # analysis_save_file = saveAnalysisFrame(pairs, spike_count_frame, bin_width, mouse_name) 
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

# if not(args.debug):
#     main()
