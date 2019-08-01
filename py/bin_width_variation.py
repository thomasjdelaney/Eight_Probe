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

def main():
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for m,mouse_name in enumerate(ep.mouse_names):
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        spon_start_time = ep.spon_start_times[m]
        cell_ids = cell_info[cell_info.mouse_name == mouse_name].index.values[:30] # using 10 cells for testing
        spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
        all_bins = ep.getAllBinsFrameForCells(cell_ids, spike_time_dict, spon_start_time)
        save_file = os.path.join(npy_dir, mouse_name + args.save_file_suffix)
        all_bins.to_pickle(save_file)
        print(dt.datetime.now().isoformat() + ' INFO: ' + save_file + ' saved.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

if not(args.debug):
    main()
