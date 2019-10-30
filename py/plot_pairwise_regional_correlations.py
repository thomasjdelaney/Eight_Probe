"""
For plotting matrices of the mean correlations and std correlations within and across regions.
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from itertools import product

parser = argparse.ArgumentParser(description='For plotting matrices of the mean correlations and std correlations within and across regions.')
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

def plotMeasureMatrix(measure_matrix, tick_labels, xlabel, suffix, bin_width, **kwargs):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.matshow(measure_matrix, cmap='Blues', **kwargs)
    ax.set_xticklabels(labels=np.hstack([[''],tick_labels]), rotation=45, fontsize='large')
    ax.set_yticklabels(labels=np.hstack([[''],tick_labels]), rotation=45, fontsize='large')
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(8.5, -0.5)
    ax.set_xlabel(xlabel, fontsize='x-large')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
    fig.colorbar(im, cax=cbar_ax)
    file_name = os.path.join(image_dir, 'regional_measure_matrices', mouse_name + '_' + str(bin_width).replace('.', 'p') + suffix + '.png')
    plt.savefig(file_name)
    plt.close()
    return file_name

def plotWithinAcrossCorr(measure_frame, mouse_name, bin_width, use_title=False):
    """
    For making a similar figure to figure 3, c from a recent paper that uses the 8 probe data.
    """
    relevant_measure_frame = measure_frame.loc[measure_frame.mouse_name == mouse_name]
    regions = relevant_measure_frame.first_region.unique()
    fig = plt.figure(figsize=(7,4))
    for first_region, second_region in zip(relevant_measure_frame.first_region, relevant_measure_frame.second_region):
        regional_measure_frame = ep.getRegionalMeasureAggFrame(relevant_measure_frame, (first_region, second_region))
        if first_region == second_region:
            plt.scatter(list(regions).index(first_region) + 1, regional_measure_frame.iloc[0]['mean_corr'], color=ep.region_to_colour[first_region])
        else:
            inds = list(regions).index(first_region) + 1, list(regions).index(second_region) + 1
            plt.scatter(inds, np.repeat(regional_measure_frame.iloc[0]['mean_corr'],2), color='black')
    plt.xticks(range(0,regions.size+2), np.hstack([[''], regions]), fontsize='x-large', rotation=30)
    plt.ylabel('Mean Corr. Coef.', fontsize='x-large')
    plt.title(mouse_name + ', Bin width=' + str(bin_width), fontsize='x-large') if use_title else None
    y_lim = plt.ylim()[1]
    plt.text(6.5, y_lim - 0.1*y_lim, 'Bin width=' + str(bin_width) + 's', fontweight='extra bold', fontsize='x-large')
    plt.tight_layout()
    file_name = os.path.join(image_dir, 'within_between_comparison', mouse_name + '_' + str(bin_width).replace('.','p') + '_corr_comp.png')
    plt.savefig(file_name)
    plt.close() 
    return file_name

if (not args.debug) & (__name__ == '__main__'):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for bin_width in ep.selected_bin_widths:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width)  + '...')
        measure_frame = ep.loadMeasureStatFile(bin_width, csv_dir)
        for mouse_name in ep.mouse_names:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
            mean_corr_matrix, regions = ep.getRegionalMeasureMatrix(measure_frame, 'mean_corr', mouse_name=mouse_name)
            mean_corr_shuff_matrix, regions = ep.getRegionalMeasureMatrix(measure_frame, 'mean_shuff_corr', mouse_name=mouse_name, regions=regions)
            file_name = plotMeasureMatrix(mean_corr_matrix, regions, 'Mean Corr. Coef', '_corr', bin_width)
            print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
            file_name = plotMeasureMatrix(mean_corr_shuff_matrix, regions, 'Mean Corr. Coef (Shuffled)', '_corr_shuff', bin_width, vmin=mean_corr_matrix.min(), vmax=mean_corr_matrix.max())
            print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
            file_name = plotWithinAcrossCorr(measure_frame, mouse_name, bin_width, use_title=False)
            print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
