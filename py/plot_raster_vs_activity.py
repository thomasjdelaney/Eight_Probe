import os, sys, argparse
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

np.random.seed(1798)
np.set_printoptions(linewidth=200)
pd.set_option('max_rows',30)

proj_dir = os.path.join(os.environ['PROJ'], 'Eight_Probe')
py_dir = os.path.join(proj_dir, 'py')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images')
mat_dir = os.path.join(proj_dir, 'mat')

sys.path.append(os.environ['PROJ'])
import Eight_Probe.py as ep

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)

mouse_name = ep.mouse_names[0]
start_time = 3900; stop_time = 3950
mouse_video = ep.loadVideoDataForMouse(mouse_name, mat_dir)
cell_ids = np.random.choice(cell_info[cell_info.mouse_name == mouse_name].index.values, 75, replace=False)
spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
video_inds = (start_time <= mouse_video['times'][0]) & (mouse_video['times'][0] < stop_time)
video_times = mouse_video['times'][0][video_inds]
max_mean_comp_ind = np.abs(mouse_video['motionSVD']).mean(axis=0).argmax()
max_mean_comp = mouse_video['motionSVD'][video_inds, max_mean_comp_ind]
ep.plotRasterWithComponent(spike_time_dict, cell_info, start_time, stop_time, max_mean_comp, video_times)
plt.show(block=False)
