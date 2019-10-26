"""
For making plots of correlation communities sorted by region, with and without the correlations shown
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from itertools import product, combinations
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser(description='For making plots of correlation communities sorted by region, with and without the correlations shown.')
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

def loadCommunityInfo(mouse_name, bin_width, npy_dir, correction='rectified', is_signal=True):
    """
    For loading in a cell_info frame with detected communities attached.
    Arguments:  mouse_name, string
                bin_width, float,
                npy_dir, string, directory
                correction, rectified or absolute
                is_signal, flag for signal or noise
    Returns:    pandas DataFrame
    """
    sig_or_noise = '_signal' if is_signal else '_noise'
    file_base_name = mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + correction + sig_or_noise + '_final_cell_info.pkl'
    return pd.read_pickle(os.path.join(npy_dir, 'communities', file_base_name))

def plotRegionalClusterMap(signal_final_cell_info):
    required_regions = signal_final_cell_info.cell_region.unique()
    num_nodes = signal_final_cell_info.shape[0]
    sorted_final_cell_info = signal_final_cell_info.sort_values(['consensus_cluster', 'cell_region', 'height'], ascending=[False, True, True])
    regional_cluster_matrix = np.zeros([num_nodes, num_nodes], dtype=int)
    for i,j in combinations(np.arange(num_nodes), 2):
        i_region = sorted_final_cell_info.iloc[i]['cell_region']
        j_region = sorted_final_cell_info.iloc[j]['cell_region']
        i_cluster = sorted_final_cell_info.iloc[i]['consensus_cluster']
        j_cluster = sorted_final_cell_info.iloc[j]['consensus_cluster']
        if (i_region == j_region) & (i_cluster == j_cluster):
            regional_cluster_matrix[i,j] = list(required_regions).index(i_region)+1
    regional_cluster_matrix = regional_cluster_matrix + regional_cluster_matrix.T
    sorted_clustering = sorted_final_cell_info['consensus_cluster'].values
    cluster_changes = np.hstack([-1, np.flatnonzero(np.diff(sorted_clustering) != 0), sorted_clustering.size-1]) + 0.5
    cmap = colors.ListedColormap(np.vstack([colors.to_rgba('gainsboro'), list(ep.region_to_colour.values())]))
    bounds = range(required_regions.size+2)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(20,21))
    ax = plt.gca()
    im = ax.matshow(regional_cluster_matrix, cmap=cmap, norm=norm)
    for i in range(cluster_changes.size-2):
        ax.plot([cluster_changes[i+1], cluster_changes[i+1]], [cluster_changes[i], cluster_changes[i+2]], color='white', linewidth=1.0)
        ax.plot([cluster_changes[i], cluster_changes[i+2]], [cluster_changes[i+1], cluster_changes[i+1]], color='white', linewidth=1.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=np.array(bounds)[1:] - 0.5)
    cb.ax.tick_params(labelsize='x-large')
    cb.set_ticklabels([c.replace('_', ' ').capitalize() for c in [''] + list(required_regions)] )
    plt.tight_layout()

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
for bin_width in ep.selected_bin_widths:
    for mouse_name in ep.mouse_names:
        signal_final_cell_info = loadCommunityInfo(mouse_name, bin_width, npy_dir)
        plotRegionalClusterMap(signal_final_cell_info)
        fig_file_name = os.path.join(image_dir, 'community_detection', 'regional_cluster_maps', mouse_name + '_' + str(bin_width).replace('.','p') + '_regional_cluster_map.png')
        plt.savefig(fig_file_name)
        print(dt.datetime.now().isoformat() + ' INFO: ' + fig_file_name + ' saved.')

# TODO  dictionary of proper names
#       make the figures bigger
