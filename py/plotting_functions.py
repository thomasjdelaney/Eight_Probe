import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gs

regions = np.array(['FrCtx','FrMoCtx','SomMoCtx','SSCtx','V1','V2','RSP','CP','LS','LH','HPF','TH','SC','MB'])
colours = cm.gist_rainbow(np.linspace(0, 1, 14)) # 14 regions
region_to_colour = dict(list(zip(regions, colours)))

def plotRaster(spike_time_dict, cell_info, start_time, stop_time):
    cell_ids = spike_time_dict.keys()
    num_cells = len(cell_ids)
    sorted_cell_info = cell_info.loc[cell_ids].sort_values(['cell_region', 'height'])
    sorted_cell_ids = sorted_cell_info.index.values
    for i, cell_id in enumerate(sorted_cell_ids):
        spike_times = spike_time_dict[cell_id][(start_time < spike_time_dict[cell_id]) & (spike_time_dict[cell_id] < stop_time)]
        if spike_times.size > 0:
            plt.vlines(x=spike_times, ymin=i+0.05, ymax=i+0.95, color=region_to_colour[sorted_cell_info.loc[cell_id]['cell_region']], alpha=1.0)
    plt.ylim([0, num_cells])
    plt.xlabel('Time (s)', fontsize='large')
    plt.xlim([start_time, stop_time])
    y_ticks = np.array([np.flatnonzero(sorted_cell_info['cell_region'] == r).mean() for r in sorted_cell_info['cell_region'].unique()])
    tick_labels = np.array([r.replace('_', ' ').capitalize() for r in sorted_cell_info['cell_region'].unique()])
    plt.yticks(y_ticks, tick_labels, fontsize='large')
    plt.tight_layout()

def plotSVDComponent(component, video_times, start_time, stop_time, use_x_axis=True):
    plt.plot(video_times, component, label='SVD component')
    plt.ylabel('Comp. Amp.', fontsize='large')
    plt.xlim([start_time, stop_time])
    if use_x_axis:
        plt.xlabel('Time (s)', fontsize='large')
    else:
        plt.xticks([])
    plt.tight_layout()

def plotRasterWithComponent(spike_time_dict, cell_info, start_time, stop_time, component, video_times):
    grid_spec = gs.GridSpec(2, 1, height_ratios=[1,4])
    plt.subplot(grid_spec[0])
    plotSVDComponent(component, video_times, start_time, stop_time, use_x_axis=False)
    plt.subplot(grid_spec[1])
    plotRaster(spike_time_dict, cell_info, start_time, stop_time)
    plt.tight_layout()

def plotMeasureHistogram(analysis_frame, measurement, x_label, y_label, x_lims=None, title=''):
    """
    For plotting the histogram of a measurement in the analysis frame. 
    """
    plt.figure(figsize=(6,5))
    plt.hist(analysis_frame[measurement], bins=21, align='left')
    plt.xlim(x_lims) if x_lims != None else None
    plt.xlabel(x_label, fontsize='x-large')
    plt.ylabel(y_label, fontsize='x-large')
    plt.title(title, fontsize='x-large') if title != '' else None
    plt.xticks(fontsize='large'); plt.yticks(fontsize='large')
