"""
A script for comparing clusters using the variation of information (VI) metric

Marina Meila, Comparing clusterings—an information based distance, Journal of Multivariate Analysis 98, 873 - 895, (2007)
Nguyen Xuan Vinh, Julien Epps, James Bailey, Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance, Journal of Machine Learning Research 11, 2837-2854, (2010)
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
from math import isclose

parser = argparse.ArgumentParser(description='For comparing clusterings using the variation of information matric (VI)')
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

def getClusteringJointDistnFromCellInfo(signal_final_cell_info):
    """
    For getting the probabilities of the joint distribution of regional and consensus clusterings from a signal_final_cell_info frame.
    Arguments:  signal_final_cell_info, DataFrame, must have 'cell_region', 'consensus_clustering' and 'mouse_name' columns
    Returns:    DataFrame, 'cell_region', 'consensus_cluster', 'joint_prob'
    """
    joint_dist_rv = signal_final_cell_info[['cell_region', 'consensus_cluster', 'mouse_name']].groupby(['cell_region', 'consensus_cluster']).count().reset_index()
    joint_dist_rv.columns = ['cell_region', 'consensus_cluster', 'joint_prob']
    joint_dist_rv.loc[:, 'joint_prob'] = joint_dist_rv.loc[:, 'joint_prob']/signal_final_cell_info.shape[0]
    assert isclose(joint_dist_rv['joint_prob'].sum(), 1), dt.datetime.now().isoformat() + ' ERROR: ' + ' Joint distribution does not sum to 1!'
    return joint_dist_rv

def buildClusteringCompMeasureDict(mouse_name, bin_width, npy_dir, correction):
    """
    For building the dictionary with clustering comparison measures for a given mouse and bin width.
    """
    signal_final_cell_info = ep.loadCommunityInfo(mouse_name, bin_width, npy_dir, correction=correction)
    signal_final_cell_info['regional_cluster'] = [list(ep.regions).index(r) for r in signal_final_cell_info['cell_region']] # replacing region strings with integers for the metric functions
    region_clustering_rv = signal_final_cell_info['cell_region'].value_counts(normalize=True)
    consensus_clustering_rv = signal_final_cell_info['consensus_cluster'].value_counts(normalize=True)
    joint_dist_rv = getClusteringJointDistnFromCellInfo(signal_final_cell_info)
    joint_entropy = entropy(joint_dist_rv.joint_prob.values)
    mi = mutual_info_score(signal_final_cell_info['regional_cluster'], signal_final_cell_info['consensus_cluster'])
    adj_mi = adjusted_mutual_info_score(signal_final_cell_info['regional_cluster'], signal_final_cell_info['consensus_cluster'])
    norm_mi = normalized_mutual_info_score(signal_final_cell_info['regional_cluster'], signal_final_cell_info['consensus_cluster'], average_method='max')
    vi = joint_entropy - mi # variation of information
    norm_vi = 1 - (mi/joint_entropy) if joint_entropy != 0 else 0
    norm_id = 1 - norm_mi # information distance
    adj_rand_ind = adjusted_rand_index(signal_final_cell_info['regional_cluster'], signal_final_cell_info['consensus_cluster'])
    return {'bin_width':bin_width, 'mouse_name':mouse_name, 'num_communities':signal_final_cell_info.consensus_cluster.max()+1, 'regional_clustering_entropy':entropy(region_clustering_rv.values), 'consensus_clustering_entropy':entropy(consensus_clustering_rv.values), 'joint_entropy':joint_entropy, 'mutual_information':mi, 'adjusted_mutual_information':adj_mi, 'normalised_mutual_information':norm_mi, 'variation_of_information':vi, 'normalised_variation_of_information':norm_vi, 'normalised_information_distance':norm_id, 'adjusted_rand_index':adj_rand_ind}

def plotClusteringCompMeasure(cluster_comp_frame, measure):
    measure_to_ylabel = {'num_communities':'Num. Communities', 'regional_clustering_entropy':r'Regional clustering $H$ (bits)', 'consensus_clustering_entropy':r'Consensus clustering $H$ (bits)', 'joint_entropy':r'Joint $H$ (bits)', 'mutual_information':'Mutual Info. (bits)', 'adjusted_mutual_information':'Adj. Mutual Info.', 'normalised_mutual_information':'Norm. Mutual Info.', 'variation_of_information':'Var. of Info. (bits)', 'normalised_variation_of_information':'Norm. Var. of Info.', 'normalised_information_distance':'Norm. Info. Distance', 'adjusted_rand_index':'Adj. Rand Index'}
    for mouse_name in ep.mouse_names:
        relevant_cluster_comp_frame = cluster_comp_frame.loc[cluster_comp_frame.mouse_name == mouse_name, ['bin_width', measure]]
        plt.plot(relevant_cluster_comp_frame.bin_width, relevant_cluster_comp_frame[measure], label=mouse_name)
    plt.legend(fontsize='large')
    plt.xlabel('Bin width (s)', fontsize='x-large')
    plt.ylabel(measure_to_ylabel[measure], fontsize='x-large')
    plt.tight_layout()
    file_name = os.path.join(image_dir, 'clustering_comparison', measure + '.png')
    plt.savefig(file_name)
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    plt.close()

if (not args.debug) & (__name__ == "__main__"):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Starting main function...')
    cluster_comp_frame = pd.DataFrame(columns=['bin_width', 'mouse_name', 'num_communities', 'regional_clustering_entropy', 'consensus_clustering_entropy', 'joint_entropy', 'mutual_information', 'adjusted_mutual_information', 'normalised_mutual_information', 'variation_of_information', 'normalised_variation_of_information', 'normalised_information_distance', 'adjusted_rand_index'])
    for bin_width in ep.selected_bin_widths:
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
        for mouse_name in ep.mouse_names:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
            measure_dict = buildClusteringCompMeasureDict(mouse_name, bin_width, npy_dir, args.correction)
            cluster_comp_frame = cluster_comp_frame.append(measure_dict, ignore_index=True)
    file_name = os.path.join(csv_dir, 'clustering_comparison', 'clustering_comparison.csv')
    cluster_comp_frame.to_csv(file_name, index=False)
    print(dt.datetime.now().isoformat() + ' INFO: ' + file_name + ' saved.')
    for measure in cluster_comp_frame.columns[2:]:
        plotClusteringCompMeasure(cluster_comp_frame, measure)
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')

