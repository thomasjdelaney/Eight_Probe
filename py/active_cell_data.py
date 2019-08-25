"""
A script for measuring how many cells are active in a given bin. Counts are grouped by brain region also. Hopefully will be useful for analysing connections between single neurons and their networks.
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt

parser = argparse.ArgumentParser(description='A script for measuring the number of active cells, and varying the bin width.')
parser.add_argument('-n', '--number_of_cells', help='Number of cells to process. Use 0 for all.', type=int, default=0)
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

if (not args.debug) & (__name__ == "__main__"):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for m,mouse_name in enumerate(ep.mouse_names):
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        spon_start_time = ep.spon_start_times[m]
        regions = cell_info.loc[cell_info.mouse_name == mouse_name, 'cell_region'].unique()
        for bin_width in ep.augmented_bin_widths:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin_width ' + str(bin_width) + '...')
            active_cell_frame = pd.DataFrame()
            for region in regions:
                print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing region ' + region + '...')
                cell_ids = cell_info.loc[(cell_info.mouse_name == mouse_name) & (cell_info.cell_region == region)].index.values
                cell_ids = cell_ids[:args.number_of_cells] if args.number_of_cells > 0 else cell_ids # selecting fewer cells for testing
                spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
                spike_count_bins = ep.getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
                active_cell_region_frame = ep.getActiveCellBinFrame(cell_ids, spike_time_dict, spike_count_bins)
                active_cell_region_frame['bin_width'] = bin_width
                active_cell_region_frame['mouse_name'] = mouse_name
                active_cell_region_frame['region'] = region
                active_cell_frame = active_cell_frame.append(active_cell_region_frame, ignore_index=True)
            active_cell_frame.to_pickle(os.path.join(npy_dir, 'active_cell_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_active.npy'))
