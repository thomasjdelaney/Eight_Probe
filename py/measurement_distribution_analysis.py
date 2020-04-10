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

def getFullRegionPairList(analysis_frame):
    data_regions = analysis_frame.first_cell_region.unique()
    comb_list = list(combinations(data_regions, 2))
    return comb_list + [(r,r) for r in data_regions]

def getAnalysisAggregationDict(region_pair, region_pair_analysis_frame):
    return {'first_region':region_pair[0], 'second_region':region_pair[1], 'num_pairs':region_pair_analysis_frame.shape[0], 
            'mean_corr':region_pair_analysis_frame.corr_coef.mean(), 'std_corr':region_pair_analysis_frame.corr_coef.std(), 
            'mean_shuff_corr':region_pair_analysis_frame.shuff_corr.mean(), 'std_shuff_corr':region_pair_analysis_frame.shuff_corr.std(), 
            'mean_info':region_pair_analysis_frame.bias_corrected_mi.mean(), 'std_info':region_pair_analysis_frame.bias_corrected_mi.std(), 
            'mean_exp_cond_cov':region_pair_analysis_frame.exp_cond_cov.mean(), 'std_exp_cond_cov':region_pair_analysis_frame.exp_cond_cov.std(),
            'mean_cov_cond_exp':region_pair_analysis_frame.cov_cond_exp.mean(), 'std_cov_conv_exp':region_pair_analysis_frame.cov_cond_exp.std(),
            'mean_exp_cond_corr':region_pair_analysis_frame.exp_cond_corr.mean(), 'std_exp_cond_corr':region_pair_analysis_frame.exp_cond_corr.std(),
            'mean_signal_corr':region_pair_analysis_frame.signal_corr.mean(), 'std_signal_corr':region_pair_analysis_frame.signal_corr.std(),
            'mean_shuff_exp_cond_cov':region_pair_analysis_frame.shuff_exp_cond_cov.mean(), 'std_shuff_exp_cond_cov':region_pair_analysis_frame.shuff_exp_cond_cov.std(),
            'mean_shuff_cov_cond_exp':region_pair_analysis_frame.shuff_cov_cond_exp.mean(), 'std_shuff_cov_cond_exp':region_pair_analysis_frame.shuff_cov_cond_exp.std(),
            'mean_shuff_exp_cond_corr':region_pair_analysis_frame.shuff_exp_cond_corr.mean(), 'std_shuff_exp_cond_corr':region_pair_analysis_frame.shuff_exp_cond_corr.std(),
            'mean_shuff_signal_corr':region_pair_analysis_frame.shuff_signal_corr.mean(), 'std_shuff_signal_corr':region_pair_analysis_frame.shuff_signal_corr.std()}

def getPlotTitle(mouse_name, num_pairs, region_pair):
    title = 'Mouse=' + mouse_name 
    if region_pair[0] == region_pair[1]:
        title = title + ', Region=' + region_pair[0]
    else:
        title = title + ', Regions=(' + region_pair[0] + ',' + region_pair[1] + ')' 
    return title + ', Num pairs=' + str(num_pairs)

def getFileName(subdir, mouse_name, bin_width, region_pair, suffix):
    dir_path = os.path.join(image_dir, subdir)
    os.mkdir(dir_path) if not os.path.isdir(dir_path) else None
    base_file_name = mouse_name + '_' + str(bin_width).replace('.', 'p') 
    if region_pair[0] == region_pair[1]:
        base_file_name = base_file_name + '_' + region_pair[0]
    else:
        base_file_name = base_file_name + '_' + region_pair[0] + '_' + region_pair[1] 
    base_file_name = base_file_name + suffix + '.png'
    return os.path.join(dir_path, base_file_name)

def plotHistogramsSave(region_pair_analysis_frame, region_pair, mouse_name, bin_width):
    plot_title = getPlotTitle(mouse_name, region_pair_analysis_frame.shape[0], region_pair)
    ep.plotMeasureHistogram(region_pair_analysis_frame, 'corr_coef', 'Corr. Coef.', 'Num. Occurances', x_lims=(-1,1), title=plot_title)
    plt.savefig(getFileName('correlation_histograms', mouse_name, bin_width, region_pair, '_corr_hist')); plt.close('all')
    ep.plotMeasureHistogram(region_pair_analysis_frame, 'shuff_corr', 'Corr. Coef. (Shuffled)', 'Num. Occurances', x_lims=(-1,1), title=plot_title)
    plt.savefig(getFileName('shuffled_correlation_histograms', mouse_name, bin_width, region_pair, '_shuff_hist')); plt.close('all')
    ep.plotMeasureHistogram(region_pair_analysis_frame, 'bias_corrected_mi', 'MI (bits)', 'Num. Occurances', title=plot_title)
    plt.savefig(getFileName('information_histograms', mouse_name, bin_width, region_pair, '_info_hist')); plt.close('all')
    ep.plotMeasureHistogram(region_pair_analysis_frame, 'exp_cond_corr', 'Event Cond. Corr.', 'Num. Occurances', title=plot_title)
    plt.savefig(getFileName('exp_cond_corr_histograms', mouse_name, bin_width, region_pair, '_exp_cond_corr')); plt.close('all')
    ep.plotMeasureHistogram(region_pair_analysis_frame, 'signal_corr', 'Signal Corr.', 'Num. Occurances', title=plot_title)
    plt.savefig(getFileName('signal_corr_histograms', mouse_name, bin_width, region_pair, '_signal_corr')); plt.close('all')

if (not args.debug) & (__name__ == '__main__'):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    measure_stats_dir = os.path.join(csv_dir, 'measure_statistics')
    os.mkdir(measure_stats_dir) if not os.path.isdir(measure_stats_dir) else None
    for bin_width in ep.selected_bin_widths:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width = ' + str(bin_width))
        measure_statistics = pd.DataFrame()
        for m,mouse_name in enumerate(ep.mouse_names):
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
            analysis_frame = ep.loadAnalysisFrame(mouse_name, bin_width, csv_dir)
            analysis_frame = ep.joinCellAnalysis(cell_info, analysis_frame)
            conditional_analysis_frame = ep.loadConditionalAnalysisFrame(mouse_name, bin_width, csv_dir)
            joined_analysis_frame = ep.joinAnalysisFrames(analysis_frame, conditional_analysis_frame)
            mouse_regional_aggregation = pd.DataFrame()
            for region_pair in getFullRegionPairList(analysis_frame):
                region_pair_analysis_frame = ep.getRegionalAnalysisFrame(joined_analysis_frame, region_pair)
                mouse_regional_aggregation = mouse_regional_aggregation.append(getAnalysisAggregationDict(region_pair, region_pair_analysis_frame), ignore_index=True)
                plotHistogramsSave(region_pair_analysis_frame, region_pair, mouse_name, bin_width)
            mouse_regional_aggregation['mouse_name'] = mouse_name
            measure_statistics = measure_statistics.append(mouse_regional_aggregation, ignore_index=True)
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving measure statistics...')
        measure_statistics.to_csv(os.path.join(measure_stats_dir, 'measure_statistics_' + str(bin_width).replace('.', 'p') + '.csv'))
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
