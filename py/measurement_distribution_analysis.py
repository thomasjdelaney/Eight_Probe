"""
For making some graphs showing the distribution of the measurements we took for selected bin widths.
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from itertools import product, combinations

parser = argparse.ArgumentParser(description='For varying the bin width used from 0.005 to 4 seconds, and taking measurements using these bin widths.')
parser.add_argument('-b', '--bin_width', help='The bin width to use.', type=float, default=2.0)
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

def plotMeasureHistogram(analysis_frame, measurement, x_label, y_label, x_lims=None, title=''):
    plt.hist(analysis_frame[measurement], bins=50)
    plt.xlim(x_lims) if x_lims != None else None
    plt.xlabel(x_label, fontsize='x-large')
    plt.ylabel(y_label, fontsize='x-large')
    plt.title(title, fontsize='x-large') if title != '' else None
    plt.xticks(fontsize='large'); plt.yticks(fontsize='large')

cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
m=1
mouse_name = ep.mouse_names[m]
analysis_frame = ep.loadAnalysisFrame(mouse_name, args.bin_width, csv_dir)
analysis_frame = analysis_frame.join(cell_info['cell_region'], how='left', on='first_cell_id')
analysis_frame = analysis_frame.rename(columns={'cell_region':'first_cell_region'})
analysis_frame = analysis_frame.join(cell_info['cell_region'], how='left', on='second_cell_id')
analysis_frame = analysis_frame.rename(columns={'cell_region':'second_cell_region'})
data_regions = analysis_frame.first_cell_region.unique().values
measure_regional_aggregation = pd.DataFrame()
for region_pair in combinations(data_regions, 2):
    region_pair_analysis_frame = analysis_frame.loc[(analysis_frame.first_cell_region == region_pair[0]) & (analysis_frame.second_cell_region == region_pair[1]) |
        (analysis_frame.first_cell_region == region_pair[1]) & (analysis_frame.second_cell_region == region_pair[0])]
    measure_regional_aggregation.append({'first_region':region_pair[0], 'second_region':region_pair[1], 'num_pairs':region_pair_analysis_frame.shape[0], 'mean_corr':region_pair_analysis_frame.corr_coef.mean(), 'std_corr':region_pair_analysis_frame.corr_coef.std(), 'mean_shuff_corr':region_pair_analysis_frame.shuff_corr.mean(), 'std_shuff_corr':region_pair_analysis_frame.shuff_corr.std(), 'mean_info':region_pair_analysis_frame.bias_corrected_mi.mean(), 'std_info':region_pair_analysis_frame.bias_corrected_mi.mstd()})

plotMeasureHistogram(analysis_frame, 'corr_coef', 'Corr. Coef.', 'Num. Occurances', x_lims=(-1,1), title='Num pairs = ' + str(analysis_frame.shape[0]))
file_name = os.path.join(image_dir, 'correlation_histograms','test_corr_hist.png')
plt.savefig(file_name)
# need to add regions 

