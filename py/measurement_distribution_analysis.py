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

def joinCellAnalysis(cell_info, analysis_frame):
    analysis_frame = analysis_frame.join(cell_info['cell_region'], how='left', on='first_cell_id')
    analysis_frame = analysis_frame.rename(columns={'cell_region':'first_cell_region'})
    analysis_frame = analysis_frame.join(cell_info['cell_region'], how='left', on='second_cell_id')
    analysis_frame = analysis_frame.rename(columns={'cell_region':'second_cell_region'})
    return analysis_frame 

def getFullRegionPairList(analysis_frame):
    data_regions = analysis_frame.first_cell_region.unique()
    comb_list = list(combinations(data_regions, 2))
    return comb_list + [(r,r) for r in data_regions]

def getRegionalAnalysisFrame(analysis_frame, region_pair):
    return analysis_frame.loc[(analysis_frame.first_cell_region == region_pair[0]) & (analysis_frame.second_cell_region == region_pair[1]) | (analysis_frame.first_cell_region == region_pair[1]) & (analysis_frame.second_cell_region == region_pair[0])]

def getAnalysisAggregationDict(region_pair_analysis_frame):
    return {'first_region':region_pair[0], 'second_region':region_pair[1], 'num_pairs':region_pair_analysis_frame.shape[0], 'mean_corr':region_pair_analysis_frame.corr_coef.mean(), 'std_corr':region_pair_analysis_frame.corr_coef.std(), 'mean_shuff_corr':region_pair_analysis_frame.shuff_corr.mean(), 'std_shuff_corr':region_pair_analysis_frame.shuff_corr.std(), 'mean_info':region_pair_analysis_frame.bias_corrected_mi.mean(), 'std_info':region_pair_analysis_frame.bias_corrected_mi.std()}

def getPlotTitle(mouse_name, num_pairs, region_pair):
    title = 'Mouse=' + mouse_name 
    if region_pair[0] == region_pair[1]:
        title = title + ', Region=' + region_pair[0]
    else:
        title = title + ', Regions=(' + region_pair[0] + ',' + region_pair[1] + ')' 
    return title + ', Num pairs=' + str(num_pairs)

def getFileName(subdir, mouse_name, bin_width, region_pair, suffix):
    base_file_name = mouse_name + '_' + str(bin_width).replace('.', 'p') 
    if region_pair[0] == region_pair[1]:
        base_file_name = base_file_name + '_' + region_pair[0]
    else:
        base_file_name = base_file_name + '_' + region_pair[0] + '_' + region_pair[1] 
    base_file_name = base_file_name + suffix + '.png'
    return os.path.join(image_dir, subdir, base_file_name)

def plotMeasureHistogram(analysis_frame, measurement, x_label, y_label, x_lims=None, title=''):
    plt.figure(figsize=(6,5))
    plt.hist(analysis_frame[measurement], bins=50)
    plt.xlim(x_lims) if x_lims != None else None
    plt.xlabel(x_label, fontsize='x-large')
    plt.ylabel(y_label, fontsize='x-large')
    plt.title(title, fontsize='x-large') if title != '' else None
    plt.xticks(fontsize='large'); plt.yticks(fontsize='large')

def plotHistogramsSave(region_pair_analysis_frame, region_pair, mouse_name, bin_width):
    plot_title = getPlotTitle(mouse_name, region_pair_analysis_frame.shape[0], region_pair)
    plotMeasureHistogram(region_pair_analysis_frame, 'corr_coef', 'Corr. Coef.', 'Num. Occurances', x_lims=(-1,1), title=plot_title)
    plt.savefig(getFileName('correlation_histograms', mouse_name, args.bin_width, region_pair, '_corr_hist')); plt.close('all')
    plotMeasureHistogram(region_pair_analysis_frame, 'shuff_corr', 'Corr. Coef. (Shuffled)', 'Num. Occurances', x_lims=(-1,1), title=plot_title)
    plt.savefig(getFileName('shuffled_correlation_histograms', mouse_name, args.bin_width, region_pair, '_shuff_hist')); plt.close('all')
    plotMeasureHistogram(region_pair_analysis_frame, 'bias_corrected_mi', 'MI (bits)', 'Num. Occurances', title=plot_title)
    plt.savefig(getFileName('information_histograms', mouse_name, args.bin_width, region_pair, '_info_hist')); plt.close('all')

if (not args.debug) & (__name__ == '__main__'):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    measure_statistics = pd.DataFrame()
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for m,mouse_name in enumerate(ep.mouse_names):
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        analysis_frame = ep.loadAnalysisFrame(mouse_name, args.bin_width, csv_dir)
        analysis_frame = joinCellAnalysis(cell_info, analysis_frame)
        mouse_regional_aggregation = pd.DataFrame()
        for region_pair in getFullRegionPairList(analysis_frame):
            region_pair_analysis_frame = getRegionalAnalysisFrame(analysis_frame, region_pair)
            mouse_regional_aggregation = mouse_regional_aggregation.append(getAnalysisAggregationDict(region_pair_analysis_frame), ignore_index=True)
            plotHistogramsSave(region_pair_analysis_frame, region_pair, mouse_name, args.bin_width)
        mouse_regional_aggregation['mouse_name'] = mouse_name
        measure_statistics = measure_statistics.append(mouse_regional_aggregation, ignore_index=True)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Saving measure statistics...')
    measure_statistics.to_csv(os.path.join(csv_dir, 'measure_statistics.csv'))
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
