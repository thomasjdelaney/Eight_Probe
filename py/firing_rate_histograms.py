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
parser.add_argument('-s', '--separate_probes', help='Flag to separate histograms from separate probes.', default=False, action='store_true')
parser.add_argument('-t', '--use_title', help='Flag to use titles on figures, or not.', default=False, action='store_true')
parser.add_argument('-a', '--use_density', help='Flag to use normalised histograms, or not.', default=False, action='store_true')
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
    
def plotFiringRateHistogram(firing_rate_region_frame, mouse_name, bin_width, region, figure_size=None, separate_probes=False, use_density=False, use_title=False):
    plt.figure(figsize=figure_size)
    bins = np.arange(0, np.ceil(firing_rate_region_frame.firing_rate.max()).astype(int) + 1)
    probe_ids = firing_rate_region_frame.probe_id.unique()
    if separate_probes:
        for probe_id in probe_ids:
            plt.hist(firing_rate_region_frame.loc[firing_rate_frame.probe_id == probe_id, 'firing_rate'].values, bins=bins, alpha=0.4, label=region + ' ' + str(probe_id), density=use_density, align='left')
    else:
        plt.hist(firing_rate_region_frame.loc[:, 'firing_rate'].values, bins=bins, label=region, density=use_density, align='left')
    plt.xlabel('Firing rate (Hz)', fontsize='x-large')
    plt.ylabel('Num. Cells', fontsize='x-large')
    plt.xticks(fontsize='large'); plt.yticks(fontsize='large')
    plt.title('Mouse name=' + mouse_name + ', bin width=' + str(bin_width) + ', total cells=' + str(firing_rate_region_frame.shape[0]), fontsize='large') if use_title else None
    plt.legend(fontsize='large')
    plt.tight_layout()
    file_name = mouse_name + '_' + region + '_' + str(bin_width).replace('.', 'p') + '_firing_rate_hist'
    file_name = file_name + '_separate' if separate_probes else file_name
    file_name = file_name + '_normalised' if use_density else file_name
    file_name = file_name + '.png'
    return file_name

if (not args.debug) & (__name__ == '__main__'):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for mouse_name in ep.mouse_names:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        firing_rate_frame = ep.loadFiringRateFrame(mouse_name, args.bin_width, npy_dir)
        firing_rate_frame = firing_rate_frame.join(cell_info, how='left', on='cell_id')
        for region in firing_rate_frame.cell_region.unique():
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing region ' + region + '...')
            firing_rate_region_frame = firing_rate_frame.loc[firing_rate_frame['cell_region'] == region, :]
            file_name = plotFiringRateHistogram(firing_rate_region_frame, mouse_name, args.bin_width, region, separate_probes=args.separate_probes, use_density=args.use_density, use_title=args.use_title)
            file_name = os.path.join(image_dir, 'firing_rate_histograms', file_name)
            plt.savefig(file_name)
            plt.close()
            print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

