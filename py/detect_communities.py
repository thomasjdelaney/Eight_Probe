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
from multiprocessing import Pool
from itertools import combinations 

parser = argparse.ArgumentParser(description='For detecting communities.')
parser.add_argument('-c', '--correction', help='Correction type. Either "absolute" or "rectified".', default='rectified', choices=['rectified', 'absolute'], type=str)
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

def getMeasureMatrix(analysis_frame, measure):
    cell_pairs = list(zip(analysis_frame.first_cell_id, analysis_frame.second_cell_id))
    cell_ids = pd.concat([analysis_frame.first_cell_id,analysis_frame.second_cell_id]).unique()
    measure_matrix = np.zeros([cell_ids.size, cell_ids.size])
    for pair, measure_value in zip(cell_pairs, analysis_frame[measure]):
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
    measure_matrix[(measure_matrix > thresholds[0]) & (measure_matrix < thresholds[1])] = 0
    return measure_matrix

if (not args.debug) & (__name__ == '__main__'):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for bin_width in ep.selected_bin_widths:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
        for mouse_name in  ep.mouse_names:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse name ' + mouse_name + '...')
            analysis_frame = ep.loadAnalysisFrame(mouse_name, bin_width, csv_dir)
            analysis_frame = ep.joinCellAnalysis(cell_info, analysis_frame)
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Calculating correlation matrix...')
            corr_matrix, cell_ids = getMeasureMatrix(analysis_frame, 'corr_coef')
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sparsifying and rectifying...')
            sparsified_matrix = sparsifyMeasureMatrix(corr_matrix.copy(), analysis_frame, 95)
            corrected_sparse_corr_matrix = rectifyMatrix(sparsified_matrix.copy()) if args.correction == 'rectified' else np.abs(sparsified_matrix.copy())
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Checking the data is symmetric...')
            corrected_sparse_corr_matrix = nnr.checkDirected(corrected_sparse_corr_matrix.copy())
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting the biggest component...')
            corrected_sparse_corr_comp, keep_indices, comp_assign, comp_size = nnr.getBiggestComponent(corrected_sparse_corr_matrix.copy())
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Sampling from null model...')
            samples_eig_vals, optional_returns = nnr.getPoissonWeightedConfModel(corrected_sparse_corr_comp, 100, return_eig_vecs=True, is_sparse=True)
            samples_eig_vecs = optional_returns['eig_vecs']
            expected_wcm = optional_returns['expected_wcm']
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing network modularity matrix...')
            network_modularity_matrix = corrected_sparse_corr_comp - expected_wcm
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Getting low dimensional space...')
            below_eig_space, below_lower_bound_inds, [mean_mins_eig, min_confidence_ints], exceeding_eig_space, exceeding_upper_bound_inds, [mean_maxs_eig, max_confidence_ints] = nnr.getLowDimSpace(network_modularity_matrix, samples_eig_vals, 0, int_type='CI')
            exceeding_space_dims = exceeding_eig_space.shape[1]
            below_space_dims = below_eig_space.shape[1]
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Splitting network into noise and signal...')
            reject_dict = nnr.nodeRejection(network_modularity_matrix, samples_eig_vals, 0, samples_eig_vecs, weight_type='linear', norm='L2', int_type='CI', bounds='upper')
            signal_weighted_adjacency_matrix = corrected_sparse_corr_comp[reject_dict['signal_inds']][:, reject_dict['signal_inds']]
            if reject_dict['signal_inds'].size == 0:
                print(dt.datetime.now().isoformat() + ' WARN: ' + 'No signal network!')
                noise_final_cell_info = cell_info.loc[cell_ids[reject_dict['noise_inds']]]
            else:
                print(dt.datetime.now().isoformat() + ' INFO: ' + 'Constructing final signal network without leaves...')
                biggest_signal_comp, biggest_signal_inds, biggest_signal_assing, biggest_signal_size = nnr.getBiggestComponent(signal_weighted_adjacency_matrix)
                signal_comp_inds = reject_dict['signal_inds'][biggest_signal_inds]
                degree_distn = (biggest_signal_comp > 0).sum(axis=0)
                leaf_inds = np.flatnonzero(degree_distn == 1)
                keep_inds = np.flatnonzero(degree_distn > 1)
                signal_final_inds = signal_comp_inds[keep_inds]
                signal_leaf_inds = signal_comp_inds[leaf_inds]
                final_weighted_adjacency_matrix = biggest_signal_comp[keep_inds][:, keep_inds]
                signal_final_cell_ids = cell_ids[signal_final_inds]
                signal_final_cell_info = cell_info.loc[signal_final_cell_ids]
                noise_final_cell_info = cell_info.loc[cell_ids[reject_dict['noise_inds']]]

                print(dt.datetime.now().isoformat() + ' INFO: ' + 'Detecting communities...')
                signal_expected_wcm = expected_wcm[signal_final_inds][:, signal_final_inds]
                max_mod_cluster, max_modularity, consensus_clustering, consensus_modularity, consensus_iterations = nnr.consensusCommunityDetect(final_weighted_adjacency_matrix, signal_expected_wcm, exceeding_space_dims+1, exceeding_space_dims+1)
                signal_final_cell_info['consensus_cluster'] = consensus_clustering
                nnr.plotClusterMap(final_weighted_adjacency_matrix, consensus_clustering, is_sort=True)
                plt.savefig(os.path.join(image_dir, 'community_detection', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + args.correction + '_cons_cluster_map.png'))
                plt.close()
                nnr.plotModEigValsVsNullEigHist(network_modularity_matrix, samples_eig_vals)
                plt.savefig(os.path.join(image_dir, 'community_detection', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + args.correction + '_eig_hist.png'))
                plt.close()
                signal_final_cell_info.to_pickle(os.path.join(npy_dir, mouse_name + str(bin_width).replace('.','p') + '_' + args.correction + '_signal_final_cell_info.pkl'))
                noise_final_cell_info.to_pickle(os.path.join(npy_dir, mouse_name + str(bin_width).replace('.','p') + '_' + args.correction + '_noise_final_cell_info.pkl'))
                print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')    
    
# TODO  move npy files into suitably named directory
#       import regional plotting function and sorting functions.
#       make the regional plots
#       recreate fig3 c of the new paper
