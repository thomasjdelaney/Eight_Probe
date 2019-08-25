"""
For comparing the active cell counts to probability distributions across bin widths, brain regions, and animals.
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import poisson, chisquare

parser = argparse.ArgumentParser(description='A script for measuring the number of active cells, and varying the bin width.')
parser.add_argument('-n', '--number_of_cells', help='Number of cells to process. Use 0 for all.', type=int, default=0)
parser.add_argument('-p', '--plot_type', help='Plot grid, single, or no plot at all.', type=str, default='No plot', choices=['No plot', 'grid', 'single'])
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

def plotActiveCellHistogram(active_cell_counts, active_bins, mouse_name, bin_width, region, num_region_cells, use_xlabel=True, use_ylabel=True):
    plt.hist(active_cell_counts, bins=active_bins, density=True)
    plt.xlabel('Num active cells', fontsize='x-large') if use_xlabel else 0
    plt.ylabel('Density', fontsize='x-large') if use_ylabel else 0
    plt.yticks(fontsize='large')
    plt.title(mouse_name + ', ' + region + ', ' + str(bin_width) + ', cells = ' + str(num_region_cells) + ', N = ' + str(active_cell_counts.size), fontsize='large')
    plt.xlim([0, active_bins[-1]])

def plotPoissonPmf(poisson_probs, active_bins):
    plt.plot(active_bins, poisson_probs, label='Poisson PMF')
    plt.legend(fontsize='large')

if (not args.debug) & (__name__ == "__main__"):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main block...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    active_cell_analysis_frame = pd.DataFrame()
    for mouse_name in ep.mouse_names:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        for bin_width in ep.augmented_bin_widths:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
            if args.plot_type == 'grid':
                fig = plt.figure(figsize=(15,15))
            active_cell_frame = ep.loadActiveCellFrame(mouse_name, bin_width, npy_dir)
            regions = active_cell_frame.region.unique()
            for r,region in enumerate(regions):
                print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing region ' + region + '...')
                num_region_cells = ((cell_info.mouse_name == mouse_name) & (cell_info.cell_region == region)).sum()
                active_cell_region_frame = active_cell_frame.loc[active_cell_frame.region == region]
                active_bins = np.arange(0, active_cell_region_frame.num_active_cells.max()+2)
                active_hist = np.histogram(active_cell_region_frame.num_active_cells, bins=active_bins, density=True)
                mean_active = active_cell_region_frame.num_active_cells.mean()
                var_active = active_cell_region_frame.num_active_cells.var()
                poisson_dist = poisson(mean_active)
                poisson_probs = poisson_dist.pmf(active_bins[:-1])
                poisson_chi_squared_stat, poisson_chi_squared_p_value = chisquare(active_hist[0], f_exp=poisson_probs)
                data_dict = {'mouse_name':mouse_name, 'bin_width':bin_width, 'region':region, 'cells_in_region':num_region_cells, 'mean_active_cells':mean_active, 'var_active_cells':var_active, 'chi_sq_stat':poisson_chi_squared_stat, 'chi_sq_p':poisson_chi_squared_p_value}
                active_cell_analysis_frame = active_cell_analysis_frame.append(data_dict, ignore_index=True)
                if args.plot_type == 'grid':
                    plt.subplot(3,3,r+1)
                    plotActiveCellHistogram(active_cell_region_frame.num_active_cells.values, active_bins, mouse_name, bin_width, region, num_region_cells, use_xlabel=r>=6, use_ylabel=0==r%3)
                    plotPoissonPmf(poisson_probs, active_bins[:-1])
                    if (r+1) == regions.size:
                        file_name = os.path.join(image_dir, 'active_cell_analysis', 'grid_' + mouse_name + '_' + region + '_' + str(bin_width).replace('.','p') + '_active.png')
                        plt.savefig(file_name)
                        plt.close()
                        print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
                elif args.plot_type == 'single':
                    plotActiveCellHistogram(active_cell_region_frame.num_active_cells.values, active_bins, mouse_name, bin_width, region, num_region_cells)
                    plotPoissonPmf(poisson_probs, active_bins[:-1])
                    file_name = os.path.join(image_dir, 'active_cell_analysis', mouse_name + '_' + region + '_' + str(bin_width).replace('.','p') + '_active.png')
                    plt.savefig(file_name)
                    plt.close()
                    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
                elif args.plot_type == 'No plot':
                    print(dt.datetime.now().isoformat() + ' INFO: ' + 'No plots.')
                else:
                    print(dt.datetime.now().isoformat() + ' WARN: ' + 'Unrecognised plot type argument ' + args.plot_type + '!')
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Bin width ' + str(bin_width) + ' complete.')
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Mouse ' + mouse_name + ' complete.')
    file_name = os.path.join(npy_dir, 'active_cell_analysis_frames', 'active_cell_analysis.npy')
    active_cell_analysis_frame.to_pickle(file_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
