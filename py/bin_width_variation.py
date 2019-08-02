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

def saveAnalysisFrame(cell_ids, spike_count_frame, bin_width, mouse_name):
    analysis_frame = ep.getAnalysisFrameForCells(cell_ids, spike_count_frame)
    analysis_frame['bin_width'] = bin_width
    save_file = os.path.join(npy_dir, 'analysis_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'analysis.npy')
    analysis_frame.to_pickle(save_file)
    return save_file

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for m,mouse_name in enumerate(ep.mouse_names):
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        spon_start_time = ep.spon_start_times[m]
        cell_ids = cell_info[cell_info.mouse_name == mouse_name].index.values[:30] # using 10 cells for testing
        spike_time_dict = ep.loadSpikeTimeDictOld(mouse_name, cell_ids, cell_info, mat_dir)
        for bin_width in ep.bin_widths:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
            spike_count_save_file, spike_count_frame = saveSpikeCountFrame(cell_ids, bin_width, spike_time_dict, spon_start_time, mouse_name)
            analysis_save_file = saveAnalysisFrame(cell_ids, spike_count_frame, bin_width, mouse_name) 
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

if not(args.debug):
    main()
