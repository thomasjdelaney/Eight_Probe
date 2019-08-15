"""
For analysing the bin width data recorded from the 3 mouse 8 probe data.
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
parser.add_argument('-n', '--number_of_cells', help='Number of cells to process. Use 0 for all.', type=int, default=10)
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

def getPositiveNegativeAnalysisFrames(analysis_frame):
    paired_frame = analysis_frame[['corr_coef', 'first_cell_id', 'second_cell_id']].groupby(['first_cell_id', 'second_cell_id']).agg(['mean', 'std', 'count'])
    pos_pairs = paired_frame[paired_frame['corr_coef']['mean'] >= 0.0].index.values
    neg_pairs = paired_frame[paired_frame['corr_coef']['mean'] < 0.0].index.values
    pos_frame = pd.concat([analysis_frame[(analysis_frame.first_cell_id == pair[0]) & (analysis_frame.second_cell_id == pair[1])]for pair in pos_pairs])
    neg_frame = pd.concat([analysis_frame[(analysis_frame.first_cell_id == pair[0]) & (analysis_frame.second_cell_id == pair[1])]for pair in neg_pairs])
    return pos_frame, neg_frame

def plotMeanCorrelationsVsBinWidth(analysis_frame, measures, measure_label, y_label, y_limits, colours=['blue', 'orange'], use_legend=True, title=''):
    agg_frame = analysis_frame[measures + ['bin_width']].groupby('bin_width').agg(['mean', 'std', 'count'])
    agg_frame.loc[:, 'measure_std_err'] = (agg_frame[measures[0]]['std']/np.sqrt(agg_frame[measures[0]]['count'])).values
    agg_frame.loc[:, 'shuff_std_err'] = (agg_frame[measures[1]]['std']/np.sqrt(agg_frame[measures[1]]['count'])).values
    plt.plot(agg_frame.index.values, agg_frame[measures[0]]['mean'].values, color=colours[0], label=r'Mean ' + measure_label)
    plt.fill_between(x=agg_frame.index.values, y1=agg_frame[measures[0]]['mean'].values + agg_frame['measure_std_err'].values, y2=agg_frame[measures[0]]['mean'].values - agg_frame['measure_std_err'].values, color=colours[0], alpha=0.25, label=r'' + measure_label + ' St. err')
    plt.plot(agg_frame.index.values, agg_frame[measures[1]]['mean'].values, color=colours[1], label=r'Mean Shuff. ' + measure_label)
    plt.fill_between(x=agg_frame.index.values, y1=agg_frame[measures[1]]['mean'].values + agg_frame['shuff_std_err'].values, y2=agg_frame[measures[1]]['mean'].values - agg_frame['shuff_std_err'].values, color=colours[1], alpha=0.25, label=r'Shuff. St. err.')
    plt.ylabel(y_label, fontsize='large')
    plt.xlabel('Bin width (s)', fontsize='large')
    plt.xlim([0.0, 4.0]);
    plt.ylim(y_limits)
    plt.xticks(fontsize='large'); plt.yticks(fontsize='large')
    plt.legend(fontsize='large') if use_legend else None
    plt.title(title, fontsize='large') if title != '' else None
    plt.tight_layout()

def saveBinWidthAnalysisFigs(mouse_name):
    analysis_frame = pd.concat([ep.loadAnalysisFrame(ep.mouse_names[0], bin_width, npy_dir) for bin_width in ep.bin_widths])
    mi_lims = [-0.05, np.ceil(analysis_frame['plugin_mi'].max())]
    pos_frame, neg_frame = getPositiveNegativeAnalysisFrames(analysis_frame)
    bin_width_figures_dir = os.path.join(image_dir, 'bin_width_analysis')
    plt.figure(figsize=(5,4))
    plotMeanCorrelationsVsBinWidth(analysis_frame, ['corr_coef', 'shuff_corr'], 'Corr.', 'Corr. Coef.', [-0.05, 1.0], title='all pairs')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_corr_all_pairs.png')
    plt.savefig(file_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    plt.figure(figsize=(5,4))
    plotMeanCorrelationsVsBinWidth(pos_frame, ['corr_coef', 'shuff_corr'], 'Corr.', 'Corr. Coef.', [-0.05, 1.0], colours=['blue', 'lightsteelblue'], title='+ve corr pairs')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_corr_pos_pairs.png')
    plt.savefig(file_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    plt.figure(figsize=(5,4))
    plotMeanCorrelationsVsBinWidth(neg_frame, ['corr_coef', 'shuff_corr'], 'Corr.', 'Corr. Coef.', [-1.0, 0.05],  colours=['green', 'lime'], title='-ve corr pairs')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_corr_neg_pairs.png')
    plt.savefig(file_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    plt.figure(figsize=(5,4))
    plotMeanCorrelationsVsBinWidth(analysis_frame, ['plugin_mi', 'plugin_shuff_mi'], 'MI', 'MI (bits)', mi_lims, title='all pairs MI')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_info_all_pairs.png')
    plt.savefig(file_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')

if (not(args.debug)) & (__name__ == '__main__'):
    [saveBinWidthAnalysisFigs(mouse_name) for mouse_name in ep.mouse_names]
