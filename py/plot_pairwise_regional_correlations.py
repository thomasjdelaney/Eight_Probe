"""
For plotting matrices of the mean correlations and std correlations within and across regions.
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from itertools import product

parser = argparse.ArgumentParser(description='For plotting matrices of the mean correlations and std correlations within and across regions.')
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

def plotMeasureMatrix(measure_matrix, tick_labels, xlabel):
    return 0

measure_frame = ep.loadMeasureStatFile(args.bin_width, csv_dir)
mouse_name = ep.mouse_names[0]
measure = 'mean_corr'
mean_corr_matrix, regions = ep.getRegionalMeasureMatrix(measure_frame, 'mean_corr', mouse_name=mouse_name)

fig, ax = plt.subplots(nrows=1, ncols=1)
im = ax.matshow(mean_corr_matrix, cmap='Blues')
ax.set_xticklabels(labels=np.hstack([[''],regions]), rotation=45, fontsize='large'); ax.set_yticklabels(labels=np.hstack([[''],regions]), rotation=45, fontsize='large')
ax.set_xlim(-0.5, 8.5); ax.set_ylim(8.5, -0.5)
ax.set_xlabel('Mean Corr. Coef', fontsize='x-large')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.11, 0.05, 0.77])
fig.colorbar(im, cax=cbar_ax)
file_name = os.path.join(image_dir, 'regional_measure_matrices', mouse_name + '_' + str(args.bin_width).replace('.', 'p') + '_corr.png')
plt.savefig(file_name)
