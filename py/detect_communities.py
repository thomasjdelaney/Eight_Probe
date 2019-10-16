"""
Script for usign the community detection method
"""
import os, sys
if float(sys.version[:3])<3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import argparse
import numpy as np  
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from itertools import combinations 

parser = argparse.ArgumentParser(description='For detecting communities.')
parser.add_argument('-p', '--percentile', help='Percentile to use when sparsifying matrix.', default=5.0, type=float)
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
import Network_Noise_Rejection_Python as nnr

def getPairMeasureValue(analysis_frame, pair, measure):
    record = analysis_frame[((analysis_frame.first_cell_id == pair[0]) & (analysis_frame.second_cell_id == pair[1])) | ((analysis_frame.first_cell_id == pair[1]) & (analysis_frame.second_cell_id == pair[0]))]
    print(dt.datetime.now().isoformat() + ' WARN: ' + 'Double record in analysis table!') if record.shape[0] > 1 else None
    return record.iloc[0][measure]

def getMeasureMatrix(analysis_frame, measure):
    cell_ids = pd.concat([analysis_frame.first_cell_id,analysis_frame.second_cell_id]).unique()
    measure_matrix = np.zeros([cell_ids.size, cell_ids.size])
    for pair in combinations(cell_ids,2):
        measure_value = getPairMeasureValue(analysis_frame, pair, measure)
        first_ind = np.flatnonzero(cell_ids == pair[0])[0]
        second_ind = np.flatnonzero(cell_ids == pair[1])[0]
        measure_matrix[first_ind, second_ind] = measure_value
        measure_matrix[second_ind, first_ind] = measure_value
    return measure_matrix, cell_ids

def rectifyMatrix(matrix):
    matrix[matrix < 0] = 0
    return matrix

def sparsifyMeasureMatrix(measure_matrix, analysis_frame, percentile=95):
    """
    Finds the 5 and 95 percentile values from the shuffled correlation column of the analysis_frame. Zeros any measure values between those two values.
    The retionale is that those values are noise and therefore should not be regarded as connections in our correlation network.
    Arguments:  measure_matrix, 
                analysis_frame,
                percentile
    Returns:    a sparser matrix
    """
    thresholds = np.percentile(analysis_frame.shuff_corr, [100 - percentile, percentile])
    measure_matrix[(measure_matrix > threshold[0]) & (measure_matrix < threshold[1])] = 0
    return measure_matrix

print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
bin_width = 1.0
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
mouse_name = ep.mouse_names[0]
print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse name ' + mouse_name + '...')
analysis_frame = ep.loadAnalysisFrame(mouse_name, bin_width, csv_dir)
analysis_frame = ep.joinCellAnalysis(cell_info, analysis_frame)
corr_matrix, cell_ids = getMeasureMatrix(analysis_frame, 'corr_coef')
sparsified_matrix = sparsifyMeasureMatrix(rectified_corr_matrix, args.percentile)
rectified_sparse_corr_matrix = rectifyMatrix(sparsified_matrix)
absolute_sparse_corr_matrix = np.abs(sparsified_matrix)

# rectified correlations
# absolute correlations
# sparsify matrix
