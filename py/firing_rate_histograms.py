"""
For creating firing rate histograms for each mouse, and each region, for the given bin width.
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='For creating firing rate histograms for each mouse, and each region, for the given bin width.')
parser.add_argument('-n', '--number_of_cells', help='Number of cells to process. Use 0 for all.', type=int, default=10)
parser.add_argument('-b', '--bin_width', help='Time bin width to use when binning spiking data.', type=float, default=2.0)
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
    
def getFiringRateFrame(mouse_name, bin_width, npy_dir):
    column_names = ['cell_id', 'spike_count_mean', 'spike_count_std', 'firing_rate', 'firing_std']
    spike_count_frame = ep.loadSpikeCountFrame(mouse_name, bin_width, npy_dir)
    agg_frame = spike_count_frame[['cell_id', 'spike_count']].groupby('cell_id').agg(['mean', 'std'])
    agg_frame = agg_frame.reset_index()
    agg_frame.columns = column_names[:3]
    agg_frame['firing_rate'] = agg_frame['spike_count_mean']/bin_width
    agg_frame['firing_std'] = agg_frame['spike_count_std']/bin_width
    return agg_frame   

def plotFiringRateHistogram(firing_rate_region_frame, mouse_name, bin_width, region, figure_size=None, use_title=False):
    plt.figure(figsize=figure_size)
    bins = np.arange(0, np.ceil(firing_rate_region_frame.firing_rate.max()).astype(int) + 1)
    probe_ids = firing_rate_region_frame.probe_id.unique()
    for probe_id in probe_ids:
        plt.hist(firing_rate_region_frame.loc[firing_rate_frame.probe_id == probe_id, 'firing_rate'].values, bins=bins, alpha=0.3)
    plt.xlabel('Firing rate (Hz)', fontsize='x-large')
    plt.ylabel('Num. Cells', fontsize='x-large')
    plt.xticks(fontsize='large'); plt.yticks(fontsize='large')
    plt.title('Mouse name=' + mouse_name + ', bin_width=' + str(bin_width) + ', total cells=' + str(firing_rate_region_frame.shape[0]), fontsize='large') if use_title else None
    return mouse_name + '_' + region + '_' + str(bin_width).replace('.', 'p') + '_firing_rate_hist.png'

if (not args.debug) & (__name__ == '__main__'):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for mouse_name in ep.mouse_names:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        firing_rate_frame = getFiringRateFrame(mouse_name, args.bin_width, npy_dir)
        firing_rate_frame = firing_rate_frame.join(cell_info, how='left')
        for region in firing_rate_frame.cell_region.unique():
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing region ' + region + '...')
            firing_rate_region_frame = firing_rate_frame.loc[firing_rate_frame['cell_region'] == region, :]
            file_name = plotFiringRateHistogram(firing_rate_region_frame, mouse_name, args.bin_width, region)
            file_name = os.path.join(image_dir, 'firing_rate_histograms', file_name)
            plt.savefig(file_name)
            print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

