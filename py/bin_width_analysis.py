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
    all_pairs = analysis_frame.loc[:,['first_cell_id', 'second_cell_id']].values
    paired_frame = analysis_frame[['corr_coef', 'first_cell_id', 'second_cell_id']].groupby(['first_cell_id', 'second_cell_id']).agg(['mean', 'std', 'count'])
    pos_pairs = paired_frame[paired_frame['corr_coef']['mean'] >= 0.0].index.values
    pos_pairs = np.array(list(map(np.array, pos_pairs)))
    neg_pairs = paired_frame[paired_frame['corr_coef']['mean'] < 0.0].index.values
    neg_pairs = np.array(list(map(np.array, neg_pairs)))
    pos_analysis_inds = np.where(np.in1d(np.ravel_multi_index(all_pairs.T, all_pairs.max(0) + 1), np.ravel_multi_index(pos_pairs.T, pos_pairs.max(0) + 1)))[0]
    neg_analysis_inds = np.where(np.in1d(np.ravel_multi_index(all_pairs.T, all_pairs.max(0) + 1), np.ravel_multi_index(neg_pairs.T, neg_pairs.max(0) + 1)))[0]
    pos_frame = analysis_frame.loc[pos_analysis_inds]
    neg_frame = analysis_frame.loc[neg_analysis_inds]
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
    plt.xlim([0.0, agg_frame.index.values.max()]);
    plt.ylim(y_limits)
    plt.xticks(fontsize='large'); plt.yticks(fontsize='large')
    plt.legend(fontsize='large') if use_legend else None
    plt.title(title, fontsize='large') if title != '' else None
    plt.tight_layout()

def plotMIVsBinWidth(analysis_frame, mi_lims, colours=['blue', 'orange', 'green'], labels=['Plugin', 'Shuffled', 'Corrected'], use_legend=True, title=''):
    agg_frame = analysis_frame[['plugin_mi', 'plugin_shuff_mi', 'bias_corrected_mi', 'bin_width']].groupby('bin_width').agg(['mean', 'std', 'count'])
    for m,measure in enumerate(['plugin_mi', 'plugin_shuff_mi', 'bias_corrected_mi']):
        agg_frame.loc[:, measure + '_std_err'] = (agg_frame[measure]['std']/np.sqrt(agg_frame[measure]['count'])).values
        plt.plot(agg_frame.index.values, agg_frame[measure]['mean'].values, color=colours[m], label=labels[m])
        plt.fill_between(x=agg_frame.index.values, y1=agg_frame[measure]['mean'].values + agg_frame[measure + '_std_err'].values, y2=agg_frame[measure]['mean'].values - agg_frame[measure + '_std_err'].values, color=colours[m], alpha=0.25)
    plt.ylabel('Mutual Information (bits)', fontsize='large')
    plt.xlabel('Bin width (s)', fontsize='large')
    plt.xlim([0.0, 4.0])
    plt.ylim(mi_lims)
    plt.xticks(fontsize='large'); plt.yticks(fontsize='large')
    plt.legend(fontsize='large') if use_legend else None
    plt.title(title, fontsize='large') if title != '' else None
    plt.tight_layout()

def plotIntraInterCorr(cell_info, analysis_frame):
    """
    For plotting mean inter-regional correlations vs mean intra-regional corrections.
    Arguments:  cell_info, pandas DataFrame
                analysis_frame, pandas DataFrame, contains the value for the correlation coefficient for each pair, for each bin width.
    Returns:    file_name, string
    """
    analysis_frame = ep.joinCellAnalysis(cell_info, analysis_frame)
    intra_regional = analysis_frame.loc[analysis_frame.first_cell_region == analysis_frame.second_cell_region]
    inter_regional = analysis_frame.loc[analysis_frame.first_cell_region != analysis_frame.second_cell_region]
    grouped_intra = intra_regional[['corr_coef','bin_width']].groupby(['bin_width']).agg(['mean','std','count'])
    grouped_inter = inter_regional[['corr_coef','bin_width']].groupby(['bin_width']).agg(['mean','std','count'])
    grouped_intra['corr_coef','std_err'] = grouped_intra['corr_coef']['std']/np.sqrt(grouped_intra['corr_coef']['count'])
    grouped_inter['corr_coef','std_err'] = grouped_inter['corr_coef']['std']/np.sqrt(grouped_inter['corr_coef']['count'])
    grouped_intra['corr_coef'].plot(y='mean', yerr='std_err', label='Intra-regional Corr.', fontsize='large')
    grouped_inter['corr_coef'].plot(y='mean', yerr='std_err', label='Inter-regional Corr.', fontsize='large')
    plt.figure(figsize=(5,4))
    plt.plot(grouped_intra.index.values, grouped_intra['corr_coef','mean'].values, label='Intra-regional Corr.', color='blue')
    plt.fill_between(grouped_intra.index.values, y1=grouped_intra['corr_coef','mean'].values - grouped_intra['corr_coef','std_err'].values, y2=grouped_intra['corr_coef','mean'].values + grouped_intra['corr_coef','std_err'].values, color='blue', alpha=0.25, label='Intra std. err.')
    plt.plot(grouped_inter.index.values, grouped_inter['corr_coef','mean'].values, label='Inter-regional Corr.', color='orange')
    plt.fill_between(grouped_inter.index.values, y1=grouped_inter['corr_coef','mean'].values - grouped_inter['corr_coef','std_err'].values, y2=grouped_inter['corr_coef','mean'].values + grouped_inter['corr_coef','std_err'].values, color='orange', alpha=0.25, label='Inter std. err.')
    plt.legend(fontsize='large')
    plt.xlabel('Time bin width (s)', fontsize='x-large'); plt.xticks(fontsize='large');
    plt.ylabel('Mean Corr. Coef.', fontsize='x-large'); plt.yticks(fontsize='large');
    plt.xlim([grouped_intra.index.values[0],grouped_intra.index.values[-1]])
    plt.tight_layout()

def saveBinWidthAnalysisFigs(mouse_name):
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    analysis_frame = pd.concat([ep.loadAnalysisFrame(mouse_name, bin_width, csv_dir) for bin_width in ep.selected_bin_widths], ignore_index=True)
    mi_lims = [-0.05, np.ceil(analysis_frame['plugin_mi'].max())]
    pos_frame, neg_frame = getPositiveNegativeAnalysisFrames(analysis_frame)
    bin_width_figures_dir = os.path.join(image_dir, 'bin_width_analysis')
    plt.figure(figsize=(5,4))
    plotMeanCorrelationsVsBinWidth(analysis_frame, ['corr_coef', 'shuff_corr'], 'Corr.', 'Corr. Coef.', [-0.05, 1.0], title='all pairs')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_corr_all_pairs.png')
    plt.savefig(file_name);plt.close();
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    plt.figure(figsize=(5,4))
    plotMeanCorrelationsVsBinWidth(pos_frame, ['corr_coef', 'shuff_corr'], 'Corr.', 'Corr. Coef.', [-0.05, 1.0], colours=['blue', 'lightsteelblue'], title='+ve corr pairs')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_corr_pos_pairs.png')
    plt.savefig(file_name);plt.close();
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    plt.figure(figsize=(5,4))
    plotMeanCorrelationsVsBinWidth(neg_frame, ['corr_coef', 'shuff_corr'], 'Corr.', 'Corr. Coef.', [-1.0, 0.05],  colours=['green', 'lime'], title='-ve corr pairs')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_corr_neg_pairs.png')
    plt.savefig(file_name);plt.close();
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    plt.figure(figsize=(5,4))
    plotMIVsBinWidth(analysis_frame, mi_lims, title='all pairs MI')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_info_all_pairs.png')
    plt.savefig(file_name);plt.close();
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    plotIntraInterCorr(cell_info, analysis_frame)
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_inter_intra_regional_correlations.png')
    plt.savefig(file_name);plt.close();
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')

if (not(args.debug)) & (__name__ == '__main__'):
    [saveBinWidthAnalysisFigs(mouse_name) for mouse_name in ep.mouse_names]
