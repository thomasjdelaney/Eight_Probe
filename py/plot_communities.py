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
parser.add_argument('-c', '--correction', help='Correction type. Either "absolute", "rectified", or "negative".', default='rectified', choices=['rectified', 'negative', 'absolute'], type=str)
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

def plotRegionalClusterMap(signal_final_cell_info, mouse_name, bin_width):
    required_regions = signal_final_cell_info.cell_region.unique()
    num_nodes = signal_final_cell_info.shape[0]
    sorted_final_cell_info = signal_final_cell_info.sort_values(['consensus_cluster', 'cell_region', 'height'], ascending=[False, True, True])
    regional_cluster_matrix = np.zeros([num_nodes, num_nodes], dtype=int)
    for i,j in combinations(np.arange(num_nodes), 2):
        i_region = sorted_final_cell_info.iloc[i]['cell_region']
        j_region = sorted_final_cell_info.iloc[j]['cell_region']
        i_cluster = sorted_final_cell_info.iloc[i]['consensus_cluster']
        j_cluster = sorted_final_cell_info.iloc[j]['consensus_cluster']
        regional_cluster_matrix[i,j] = list(required_regions).index(i_region)+1 if (i_region == j_region) & (i_cluster == j_cluster) else 0
    regional_cluster_matrix = regional_cluster_matrix + regional_cluster_matrix.T
    for i in range(num_nodes):
        i_region = sorted_final_cell_info.iloc[i]['cell_region']
        regional_cluster_matrix[i,i] = list(required_regions).index(i_region)+1
    sorted_clustering = sorted_final_cell_info['consensus_cluster'].values
    cluster_changes = np.hstack([-0.5, np.flatnonzero(np.diff(sorted_clustering) != 0), sorted_clustering.size-0.5])
    cmap = colors.ListedColormap(np.vstack([colors.to_rgba('gainsboro'), list([ep.region_to_colour[region] for region in required_regions])]))
    bounds = range(required_regions.size+2)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(6.6, 4.8))
    ax = plt.gca()
    im = ax.matshow(regional_cluster_matrix, cmap=cmap, norm=norm)
    x_lim, y_lim = ax.get_xlim()[1], ax.get_ylim()[0]
    ax.text(0.6*x_lim, -0.05 + 0.1*y_lim, 'Bin width=' + str(bin_width) + 's', fontweight='extra bold', fontsize='large')
    for i in range(cluster_changes.size-2):
        ax.plot([cluster_changes[i+1], cluster_changes[i+1]], [cluster_changes[i], cluster_changes[i+2]], color='white', linewidth=1.0)
        ax.plot([cluster_changes[i], cluster_changes[i+2]], [cluster_changes[i+1], cluster_changes[i+1]], color='white', linewidth=1.0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=np.array(bounds)[1:] - 0.5)
    cb.ax.tick_params(labelsize='x-large')
    cb.set_ticklabels([c.replace('_', ' ').capitalize() for c in ['mixed regions'] + list(required_regions)] )
    plt.tight_layout()

if (not args.debug) & (__name__ == "__main__"):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    for bin_width in ep.selected_bin_widths:
        for mouse_name in ep.mouse_names:
            signal_final_cell_info = ep.loadCommunityInfo(mouse_name, bin_width, npy_dir, correction=args.correction)
            plotRegionalClusterMap(signal_final_cell_info, mouse_name, bin_width)
            fig_file_name = os.path.join(image_dir, 'community_detection', 'regional_cluster_maps', args.correction, mouse_name + '_' + str(bin_width).replace('.','p') +'_regional_cluster_map.png')
            plt.savefig(fig_file_name)
            print(dt.datetime.now().isoformat() + ' INFO: ' + fig_file_name + ' saved.')

# TODO  dictionary of proper names
#       make the figures bigger
