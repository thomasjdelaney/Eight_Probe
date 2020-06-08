"""
For comparing spike count distributions to Poisson and Gaussian Distributions.
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from itertools import product, combinations
from scipy.stats import poisson, norm, chisquare

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

def getRandomSelection(spike_count_frame, num_cells=10):
    return np.random.choice(spike_count_frame.cell_id.unique(), size=num_cells, replace=False)

def plotHistDists(spike_counts, firing_rate, firing_std):
    bins=np.arange(spike_counts.max()+1)
    plt.hist(spike_counts, bins=bins, density=True, label='Spike count dist')
    poiss_dist = poisson(firing_rate)
    plt.plot(bins, poiss_dist.pmf(bins), label='Poisson dist')
    gauss_dist = norm(firing_rate, firing_std)
    plt.plot(bins, gauss_dist.pdf(bins), label='Gaussian dist')
    plt.xlim([bins[0],bins[-1]])
    plt.legend(); plt.tight_layout()

def plotSpikeCountAnalysis(bin_width_agg_frame):
    rate_frame = bin_width_agg_frame[['bin_width', 'firing_rate']].groupby('bin_width').agg(['mean', 'std'])
    rate_frame.reset_index(inplace=True)
    rate_frame.columns = ['bin_width', 'mean_firing_rate', 'std_firing_rate']
    std_frame = bin_width_agg_frame[['bin_width', 'firing_std']].groupby('bin_width').agg(['mean', 'std'])
    std_frame.reset_index(inplace=True)
    std_frame.columns = ['bin_width', 'mean_firing_std', 'std_firing_std']
    plt.figure()
    plt.plot(rate_frame.bin_width, rate_frame.mean_firing_rate, label='Mean firing rate', color='blue')
    plt.fill_between(x=rate_frame.bin_width, y1=rate_frame.mean_firing_rate + std_frame.mean_firing_std, y2=rate_frame.mean_firing_rate - std_frame.mean_firing_std, alpha=0.25, color='blue', label='st. dev.')
    plt.xlabel('Bin width (s)'); plt.ylabel('Firing rate (Hz)')
    plt.ylim([0,rate_frame.mean_firing_rate.max() + std_frame.mean_firing_std.max()])
    plt.xlim([0, 4.0])
    plt.legend()
    plt.tight_layout()

def plotActiveCells(npy_dir):
    for i, mouse_name in enumerate(ep.mouse_names):
        plt.subplot(ep.mouse_names.shape[0], 1, i+1)
        spike_count_frame = ep.loadSpikeCountFrame(mouse_name, 0.01, npy_dir)
        spike_count_array = np.array([spike_count_frame.loc[spike_count_frame.cell_id == cell_id, 'spike_count'] for cell_id in spike_count_frame.cell_id.unique()])
        active_cells = spike_count_array.astype(bool).sum(axis=0)
        plt.plot(spike_count_frame.bin_start_time.unique(), active_cells)
        plt.xlabel('Time (s)'); plt.ylabel('Number of active cells')
    plt.tight_layout()

def plotMovingAverages(spike_count_frame, num_cells=10, window_size=10):
    cell_ids = getRandomSelection(spike_count_frame, num_cells)
    spike_count_frame['spike_count_moving_average'] = spike_count_frame[['cell_id', 'spike_count']].groupby('cell_id')['spike_count'].transform(lambda x: x.rolling(10,1).mean())
    spike_count_frame['firing_rate_moving_average'] = spike_count_frame['spike_count_moving_average']/spike_count_frame.bin_width.unique()[0]
    bin_times = spike_count_frame.bin_start_time.unique()
    for cell_id in cell_ids:
        plt.plot(bin_times, spike_count_frame.loc[spike_count_frame.cell_id == cell_id, 'firing_rate_moving_average'], alpha=0.3)
    plt.xlabel('Time (s)'); plt.ylabel('Firing Rate (Hz)');
    plt.tight_layout()

def plotChiSquaredStats(bin_width_agg_frame):
    bin_width_agg_frame['poiss_chi_squared_stat'] = np.log10(bin_width_agg_frame['poiss_chi_squared_stat'])
    bin_width_agg_frame['gaussian_chi_squared_stat'] = np.log10(bin_width_agg_frame['gaussian_chi_squared_stat'])
    for mouse_name in ep.mouse_names:
        agg = bin_width_agg_frame.loc[bin_width_agg_frame.mouse_name == mouse_name]
        rate_frame = agg[['bin_width', 'poiss_chi_squared_stat', 'gaussian_chi_squared_stat']].groupby('bin_width').agg(['mean', 'std', 'count'])
        rate_frame.reset_index(inplace=True)
        rate_frame.columns = ['bin_width', 'poiss_chi_squared_stat_mean', 'poiss_chi_squared_stat_std', 'num_samples', 'gaussian_chi_squared_stat_mean', 'gaussian_chi_squared_stat_std', 'ns']
        rate_frame['poiss_chi_squared_std_err'] = rate_frame.poiss_chi_squared_stat_std/np.sqrt(rate_frame.num_samples)
        rate_frame['gaussian_chi_squared_std_err'] = rate_frame.gaussian_chi_squared_stat_std/np.sqrt(rate_frame.num_samples)
        plt.figure(figsize=(6,4))
        plt.plot(rate_frame.bin_width, rate_frame.poiss_chi_squared_stat_mean, color='blue', label=r'mean Poiss. $\chi^2$ stat.')
        plt.fill_between(x=rate_frame.bin_width, y1=rate_frame.poiss_chi_squared_stat_mean - rate_frame.poiss_chi_squared_std_err, y2=rate_frame.poiss_chi_squared_stat_mean + rate_frame.poiss_chi_squared_std_err, color='blue', alpha=0.25, label='std. err. Poiss. $\chi^2$ stat')
        plt.plot(rate_frame.bin_width, rate_frame.gaussian_chi_squared_stat_mean, color='orange', label='mean Gauss. $\chi^2$ stat.')
        plt.fill_between(x=rate_frame.bin_width, y1=rate_frame.gaussian_chi_squared_stat_mean - rate_frame.gaussian_chi_squared_std_err, y2=rate_frame.gaussian_chi_squared_stat_mean + rate_frame.gaussian_chi_squared_std_err, color='orange', alpha=0.25, label='std. err. Gauss. $\chi^2$ stat.')
        plt.xlabel('Bin width (s)', fontsize='x-large')
        plt.ylabel(r'$\log _{10} \chi^2$ Stat.', fontsize='x-large')
        plt.legend(fontsize='large')
        plt.xticks(fontsize='large');plt.yticks(fontsize='large');
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'bin_width_analysis', mouse_name + '_stats_by_bin_width.png'))
        plt.close()

def loadBinWidthAggregate(npy_dir):
    column_names = ['bin_width', 'cell_id', 'spike_count_mean', 'spike_count_std', 'mouse_name', 'firing_rate', 'firing_std', 'poiss_chi_squared_stat', 'gaussian_chi_squared_stat']
    bin_width_agg_frame = pd.DataFrame(columns=column_names)
    for mouse_name in ep.mouse_names:
        for bin_width in ep.bin_widths:
            spike_count_frame = ep.loadSpikeCountFrame(mouse_name, bin_width, npy_dir)
            agg_frame = spike_count_frame[['cell_id', 'spike_count', 'bin_width']].groupby(['bin_width', 'cell_id']).agg(['mean', 'std'])
            agg_frame = agg_frame.reset_index()
            agg_frame.columns = column_names[:-5]
            agg_frame['mouse_name'] = mouse_name
            agg_frame['firing_rate'] = agg_frame['spike_count_mean']/bin_width
            agg_frame['firing_std'] = agg_frame['spike_count_std']/bin_width
            for cell_id in agg_frame.cell_id.unique():
                spike_counts = spike_count_frame.loc[spike_count_frame.cell_id == cell_id, 'spike_count'].values
                spike_bins = np.arange(0, spike_counts.max()+2)
                spike_count_hist = np.histogram(spike_counts, bins=spike_bins, density=True)
                poiss_dist = poisson(spike_counts.mean() / bin_width)
                gaussian_dist = norm(spike_counts.mean() / bin_width, spike_counts.std() / bin_width)
                poiss_chi_squared_test = chisquare(spike_count_hist[0], f_exp=poiss_dist.pmf(spike_bins[:-1]))
                gaussian_chi_squared_test = chisquare(spike_count_hist[0], f_exp=gaussian_dist.pdf(spike_bins[:-1]))
                agg_frame.loc[agg_frame.cell_id == cell_id, 'poiss_chi_squared_stat'] = poiss_chi_squared_test.statistic
                agg_frame.loc[agg_frame.cell_id == cell_id, 'gaussian_chi_squared_stat'] = gaussian_chi_squared_test.statistic
            bin_width_agg_frame = bin_width_agg_frame.append(agg_frame, ignore_index=True)
    return bin_width_agg_frame

if not args.debug:
    agg = loadBinWidthAggregate(npy_dir)
    plotChiSquaredStats(agg)
