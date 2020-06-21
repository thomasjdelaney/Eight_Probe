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

def plotTwoMeasureCondComp(mouse_name, first_measure, second_measure, first_label, second_label, title=''):
    """
    For plotting the average expected condition correlation and the average correlation of conditional expectations over bin width.
    Arguments:  mouse_name, string
                first_measure, str, example:'exp_cond_corr'
                second_measure, str, example:'signal_corr'
                first_label, str, example:r'$\rho_{X,Y|Z}$'
                second_label, str, example:r'$\rho_{signal}$'
                title, str, example:'Mouse ' + str(list(ep.mouse_names).index(mouse_name)) + ', cond. corr. measures'
    Returns:    Nothing
    """
    cond_analysis_frame = pd.concat([ep.loadConditionalAnalysisFrame(mouse_name, bin_width, csv_dir) for bin_width in ep.selected_bin_widths], ignore_index=True)
    first_means = np.zeros(ep.selected_bin_widths.size)
    first_std = np.zeros(ep.selected_bin_widths.size)
    first_counts = np.zeros(ep.selected_bin_widths.size)
    second_means = np.zeros(ep.selected_bin_widths.size)
    second_std = np.zeros(ep.selected_bin_widths.size)
    second_counts = np.zeros(ep.selected_bin_widths.size)
    for i,bin_width in enumerate(ep.selected_bin_widths):
        width_frame = cond_analysis_frame[cond_analysis_frame['bin_width'] == bin_width]
        first_means[i] = np.nanmean(width_frame[first_measure])
        first_std[i] = np.nanstd(width_frame[first_measure])
        first_counts[i] = np.isfinite(width_frame[first_measure]).sum()
        second_means[i] = np.nanmean(width_frame[second_measure])
        second_std[i] = np.nanstd(width_frame[second_measure])
        second_counts[i] = np.isfinite(width_frame[second_measure]).sum()
    first_std_errs = first_std/np.sqrt(first_counts)
    second_std_errs = second_std/np.sqrt(second_counts)
    plt.plot(ep.selected_bin_widths, first_means, label=first_label, color='blue')
    plt.fill_between(x=ep.selected_bin_widths, y1=first_means-first_std_errs, y2=first_means+first_std_errs, color='blue',alpha=0.25)
    plt.plot(ep.selected_bin_widths, second_means, label=second_label, color='orange')
    plt.fill_between(x=ep.selected_bin_widths, y1=second_means-second_std_errs, y2=second_means+second_std_errs, color='orange',alpha=0.25)
    plt.xlabel('Time bin width (s)',fontsize='x-large');plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    plt.xlim([ep.selected_bin_widths[0], ep.selected_bin_widths[-1]])
    plt.title(title, fontsize='x-large') if title != '' else None
    plt.legend(fontsize='large')
    plt.tight_layout()

def getGoodPair(analysis_frame, mouse_name, cell_info, is_inter=False):
    """
    For getting a nice pair for an example.
    Arguments:  analysis_frame, pandas DataFrame,
                is_inter, boolean, do we want an inter-regional pair?
    Returns: pair, 2 ints
    """
    if is_inter:
        pairs = analysis_frame.loc[(analysis_frame.first_cell_region != analysis_frame.second_cell_region)&(analysis_frame.bin_width == 0.005), ['first_cell_id','second_cell_id']].values
    else:
        pairs = analysis_frame.loc[(analysis_frame.first_cell_region == analysis_frame.second_cell_region)&(analysis_frame.bin_width == 0.005), ['first_cell_id','second_cell_id']].values
    num_pairs = pairs.shape[0]
    have_pair = False
    while not have_pair:
        pair = pairs[np.random.choice(range(num_pairs))]
        pair_frame = analysis_frame[(analysis_frame.first_cell_id == pair[0])&(analysis_frame.second_cell_id == pair[1])]
        spike_time_dict = ep.loadSpikeTimeDict(mouse_name, pair, cell_info, mat_dir)
        ok_num_spikes = (100 < spike_time_dict.get(pair[0]).size) & (spike_time_dict.get(pair[0]).size < 1000) & (100 < spike_time_dict.get(pair[1]).size) & (spike_time_dict.get(pair[1]).size < 1000)
        has_all_bins = np.all(pair_frame.bin_width.unique() == ep.selected_bin_widths)
        is_increasing = np.all(pair_frame.corr_coef.diff()[1:]>0)
        have_pair = has_all_bins & is_increasing & ok_num_spikes
    return pair

def plotPairsRaster(spike_time_dict, good_intra_pair, good_inter_pair):
    """
    For plotting a raster plot of the 4 cells in the inter and intra pairs. Colour accordingly.
    Arguments:  spike_time_dict, dict, cell_id => spike times
                good_intra_pair, 2 ints
                good_inter_pair, 2 ints
    Returns:    nothing
    """
    plt.figure(figsize=(5,4))
    plt.vlines(x=spike_time_dict.get(good_inter_pair[0]), ymin=0.05, ymax=0.95, color='orange')
    plt.vlines(x=spike_time_dict.get(good_inter_pair[1]), ymin=1+0.05, ymax=1+0.95, color='orange')
    plt.vlines(x=spike_time_dict.get(good_intra_pair[0]), ymin=2+0.05, ymax=2+0.95, color='blue')
    plt.vlines(x=spike_time_dict.get(good_intra_pair[1]), ymin=3+0.05, ymax=3+0.95, color='blue')
    spike_times = np.concatenate(list(spike_time_dict.values()))
    plt.xlim([spike_times.min(), spike_times.max()])
    plt.yticks([])
    plt.xticks(fontsize='large')
    plt.xlabel('Time (s)', fontsize='x-large')
    plt.tight_layout()

def plotPairsCorrs(intra_pair_frame, inter_pair_frame):
    """
    For plotting the correlations of the inter pair and the intra pair.
    Arguments:  intra_pair_frame,
                inter_pair_frame,
    Returns:    nothing
    """
    plt.figure(figsize=(5,4))
    plt.plot(inter_pair_frame.bin_width, inter_pair_frame.corr_coef, color='orange',label='Inter reg. pair')
    plt.plot(intra_pair_frame.bin_width, intra_pair_frame.corr_coef, color='blue',label='Intra reg. pair')
    plt.xlim([ep.selected_bin_widths[0], ep.selected_bin_widths[-1]])
    plt.xlabel('Time bin width (s)', fontsize='x-large')
    plt.ylabel('Corr. Coef.', fontsize='x-large')
    plt.xticks(fontsize='large');plt.yticks(fontsize='large')
    plt.legend(fontsize='large')
    plt.tight_layout()

def plotExamplePairs(mouse_name, cell_info, analysis_frame):
    """
    For plotting the correlations between a pair of neurons across bin widths, for a within and a between pair.
    Maybe also plot the spike counts, or some info on the firing rate.
    """
    analysis_frame = ep.joinCellAnalysis(cell_info, analysis_frame)
    good_intra_pair = getGoodPair(analysis_frame, mouse_name, cell_info)
    good_inter_pair = getGoodPair(analysis_frame, mouse_name, cell_info, is_inter=True)
    intra_pair_frame = analysis_frame[(analysis_frame.first_cell_id == good_intra_pair[0])&(analysis_frame.second_cell_id == good_intra_pair[1])]
    inter_pair_frame = analysis_frame[(analysis_frame.first_cell_id == good_inter_pair[0])&(analysis_frame.second_cell_id == good_inter_pair[1])]
    spike_time_dict = ep.loadSpikeTimeDict(mouse_name, np.concatenate([good_intra_pair, good_inter_pair]), cell_info, mat_dir)
    plotPairsRaster(spike_time_dict, good_intra_pair, good_inter_pair)
    save_name = os.path.join(image_dir, 'pair_analysis', mouse_name, 'pairs_raster.png')
    dir_name = os.path.dirname(save_name)
    os.makedirs(dir_name) if not(os.path.isdir(dir_name)) else None
    plt.savefig(save_name);plt.close('all');
    plotPairsCorrs(intra_pair_frame, inter_pair_frame)
    save_name = os.path.join(image_dir, 'pair_analysis', mouse_name, 'pairs_correlation.png')
    dir_name = os.path.dirname(save_name)
    os.makedirs(dir_name) if not(os.path.isdir(dir_name)) else None
    plt.savefig(save_name);plt.close('all');
    return dir_name

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
    plt.figure(figsize=(4,3))
    plotTwoMeasureCondComp(mouse_name, 'exp_cond_corr', 'signal_corr', r'$\rho_{X,Y|Z}$', r'$\rho_{signal}$', title='Mouse ' + str(list(ep.mouse_names).index(mouse_name)) + ', cond. corr. measures')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_cond_corr_comparison.png')
    plt.savefig(file_name);plt.close()
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    plt.figure(figsize=(4,3))
    plotTwoMeasureCondComp(mouse_name, 'exp_cond_cov', 'cov_cond_exp', r'E[Cov($X,Y|Z$)]', r'Cov(E[$X|Z$], E[$Y|Z$])', title='Mouse ' + str(list(ep.mouse_names).index(mouse_name)) + ', cond. cov. measures')
    file_name = os.path.join(bin_width_figures_dir, mouse_name + '_cond_cov_comparison.png')
    plt.savefig(file_name);plt.close()
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    dir_name = plotExamplePairs(mouse_name, cell_info, pos_frame)
    print(dt.datetime.now().isoformat() + ' INFO: ' + dir_name + ' saved.')

if (not(args.debug)) & (__name__ == '__main__'):
    [saveBinWidthAnalysisFigs(mouse_name) for mouse_name in ep.mouse_names]
