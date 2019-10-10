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

def plotMeasureHistogram(analysis_frame, measurement, x_label, y_label, x_lims=None, title=''):
    plt.hist(analysis_frame[measurement], bins=50)
    plt.xlim(x_lims) if x_lims != None else None
    plt.xlabel(x_label, fontsize='x-large')
    plt.ylabel(y_label, fontsize='x-large')
    plt.title(title, fontsize='x-large') if title != '' else None
    plt.xticks(fontsize='large'); plt.yticks(fontsize='large')

m=0
mouse_name = ep.mouse_names[m]
analysis_frame = ep.loadAnalysisFrame(mouse_name, args.bin_width, csv_dir)

plotMeasureHistogram(analysis_frame, 'corr_coef', 'Corr. Coef.', 'Num. Occurances', x_lims=(-1,1), title='Num pairs = ' + str(analysis_frame.shape[0]))
file_name = os.path.join(image_dir, 'correlation_histograms','test_corr_hist.png')
plt.savefig(file_name)
# need to add regions 

