"""
For varying the bin width used from 0.005 to 2 seconds, and taking measurements using these bin widths.

We save a frame for spike counts for each time bin containing all cells. We also save a frame for each time bin for all pairs.
"""
import os, argparse, sys, shutil, warnings
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import pandas as pd
import datetime as dt
from itertools import product, combinations
from scoop import futures
from multiprocessing import Pool
from functools import reduce
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV
from statsmodels.stats.stattools import durbin_watson
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

parser = argparse.ArgumentParser(description='For varying the bin width used from 0.005 to 4 seconds, and taking measurements using these bin widths.')
parser.add_argument('-n', '--number_of_cells', help='Number of cells to process. Use 0 for all.', type=int, default=10)
parser.add_argument('-f', '--save_firing_rate_frame', help='Flag to indicate whether or not firing rates should be saved.', default=False, action='store_true')
parser.add_argument('-a', '--save_analysis_frame', help='Flag to indicate whether or not analysis should be performed and saved.', default=False, action='store_true')
parser.add_argument('-z', '--save_conditional_correlations', help='Flag to indicate whether or not to calculate and save conditional correlations.', default=False, action='store_true')
parser.add_argument('-s', '--save_spike_count_frame', help='Flag to indicate wether or not save the spike count frames.', default=False, action='store_true')
parser.add_argument('-c', '--num_chunks', help='Number of chunks to split the pairs into before processing.', default=10, type=int)
parser.add_argument('-p', '--num_components', help='Number of principle components to condition upon.', default=1, type=int)
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

pd.set_option('max_rows',30) # setting display options for terminal display
np.set_printoptions(linewidth=shutil.get_terminal_size().columns)

proj_dir = os.path.join(os.environ['PROJ'], 'Eight_Probe')
py_dir = os.path.join(proj_dir, 'py')
npy_dir = os.path.join(proj_dir, 'npy')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images')
mat_dir = os.path.join(proj_dir, 'mat')

sys.path.append(os.environ['PROJ'])
import Eight_Probe.py as ep

def constructMapFuncArgs(pairs, spike_count_dict):
    """
    For constructing a list of dictionaries to be passed into a mapping function for scoop.futures.mapReduce.
    In this context the mapping function can only take one argument.
    Arguments:  pairs, numpy.array, all the possible pairs.
                spike_count_dict, cell_id => spike counts
    Returns:    List of dictionaries, each with two keys, values are arrays of spike counts.
    """
    dict_list = list([])
    for pair in pairs:
        dict_list.append({pair[0]:spike_count_dict[pair[0]], pair[1]:spike_count_dict[pair[1]]})
    return dict_list

def constructMapFuncArgsOld(pairs, spike_count_frame):
    """
    For constructing a list of dictionaries to be passed into a mapping function for scoop.futures.mapReduce.
    In this context the mapping function can only take one argument.
    Arguments:  pairs, numpy.array, all the possible pairs.
                spike_count_frame, DataFrame, cell_id, spike_count, bin_start_time, bin_stop_time
    """
    return [{pair[0]:spike_count_frame.loc[spike_count_frame.cell_id == pair[0], 'spike_count'].values, pair[1]:spike_count_frame.loc[spike_count_frame.cell_id == pair[1], 'spike_count'].values} for pair in pairs]

def saveSpikeCountFrame(cell_ids, bin_width, spike_time_dict, spon_start_time, mouse_name):
    """
    For getting, saving, and returning the spike count frame, with the bin width column. Also returns the name of the file where the spike_count_bins was saved.
    """
    spike_count_bins = ep.getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time)
    spike_count_frame = ep.getSpikeCountBinFrame(cell_ids, spike_time_dict, spike_count_bins)
    spike_count_frame['bin_width'] = bin_width
    save_file = os.path.join(npy_dir, 'spike_count_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'spike_counts.npy')
    spike_count_frame.to_pickle(save_file)
    return save_file, spike_count_frame

def getAnalysisDictForPair(pair_count_dict):
    """
    For getting a dictionary containing measurements for the given pair. This function is most useful for parallel processing
    as there will be a great number of pairs. This is the mapping function for mapReduce.
    Arguments:  pair_count_dict, int => numpy array int, the two keys is the pair, the values are the spike counts
    Returns:    Dict,   keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv, bias_corrected_mi)
    """
    pair = np.array(list(pair_count_dict.keys()))
    corr, corr_pv, shuff_corr, shuff_corr_pv = ep.getSpikeCountCorrelationsForPair(pair_count_dict)
    plugin_mi, plugin_shuff_mi, bias_corrected_mi = ep.getMutualInfoForPair(pair_count_dict)
    return {'corr_coef': np.repeat(corr,1), 'corr_pv':np.repeat(corr_pv,1), 'first_cell_id':np.repeat(pair[0],1), 'plugin_mi':np.repeat(plugin_mi,1), 'plugin_shuff_mi':np.repeat(plugin_shuff_mi,1), 'second_cell_id':np.repeat(pair[1],1), 'shuff_corr':np.repeat(shuff_corr,1), 'shuff_corr_pv':np.repeat(shuff_corr_pv,1), 'bias_corrected_mi':np.repeat(bias_corrected_mi,1)}

def getConditionalExpectation(spike_count_dict, time_bins, svd_comp, svd_times, num_bins_svd=50):
    """
    For calculating the conditional expectation of the spike counts given num_bins_svd different binned values of svd_comp.
    Arguments:  spike_count_dict, dict cell_id => spike_counts
                time_bins, numpy array (float), times for the spike counts.
                svd_comp, numpy array (float), singular value decomposition component
                svd_times, numpy array (float), times for the svd measurements
    Returns:    svd_marginal_distn, P(Z_i = z_i)
                conditional_expectation_dict, cell_id => conditional expectation of the spike counts given the svd comp, numpy array (float) of length = num_bins_svd

    NB check this again.
    """
    svd_counts, svd_bins = np.histogram(svd_comp, bins=num_bins_svd)
    svd_marginal_distn = svd_counts / svd_counts.sum()
    svd_value_bins = np.digitize(svd_comp, svd_bins, right=True)-1
    svd_value_bins[svd_value_bins == svd_value_bins.min()] = 0
    conditional_expectation_dict = {}
    for cell_id, spike_counts in spike_count_dict.items():
        spike_count_list = list(range(spike_counts.min(), spike_counts.max()+1)) # list for faster indexing
        joint_distn = np.zeros((len(spike_count_list), num_bins_svd), dtype=float)
        for i in range(svd_counts.size):
            svd_bin_value_times = svd_times[np.nonzero(svd_value_bins == i)[0]]
            if svd_bin_value_times.size > 0:
                svd_bin_value_time_bin_inds = np.digitize(svd_bin_value_times, time_bins, right=True)
                svd_bin_value_spike_count_values, svd_bin_value_spike_count_counts = np.unique(spike_counts[svd_bin_value_time_bin_inds-1], return_counts=True)
                joint_distn[[spike_count_list.index(spikes) for spikes in svd_bin_value_spike_count_values], i] += svd_bin_value_spike_count_counts
        joint_distn = joint_distn / joint_distn.sum()
        with np.errstate(divide='ignore', invalid='ignore'): # avoiding warning messages
            cond_distn = joint_distn / svd_marginal_distn
        cond_distn[np.isnan(cond_distn)] = 0.0
        conditional_expectation_dict[cell_id] = np.dot(np.array(spike_count_list), cond_distn)
    return svd_marginal_distn, conditional_expectation_dict

def getCompExpCondCov(svd_comps, svd_times, spike_count_dict, time_bins, num_bins_svd=50):
    """
    For getting the expected conditional covariance for each component individually.
    Arguments:  svd_comps, singular value decomposition components
                svd_times, time stamps for component measures
                spike_count_dict, Dict, cell_id => spike counts
                time_bins, the spike count time bin borders,
    Returns:    comp_expected_cond_cov,
    """
    num_comps = svd_comps.shape[1]
    num_cells = len(spike_count_dict)
    spike_count_array = np.array(list(spike_count_dict.values()))
    mean_of_products_of_spike_counts = np.array([np.outer(s, s) for s in spike_count_array.T]).mean(axis=0)
    with Pool() as pool:
        cond_exp_futures = pool.starmap_async(getConditionalExpectation, zip(num_comps*[spike_count_dict], num_comps*[time_bins], svd_comps.T, num_comps*[svd_times]))
        cond_exp_futures.wait()
    cond_exp_got = cond_exp_futures.get()
    comp_expected_cond_cov = np.empty(shape=(num_comps, num_cells, num_cells), dtype=float)
    for i in range(num_comps):
        svd_marginal_dist = cond_exp_got[i][0]
        cond_expected_array = np.array(list(cond_exp_got[i][1].values()))
        comp_expected_cond_cov[i, :, :] = np.tensordot(svd_marginal_dist, np.array([np.outer(s, s) for s in cond_expected_array.T]), axes=(0,0))
    comp_expected_cond_cov = mean_of_products_of_spike_counts - comp_expected_cond_cov
    return comp_expected_cond_cov

def getCompsRankedByExpCondCov(comp_expected_cond_cov, num_comps_returned=4):
    """
    For getting some kind of sort order of the components by the expected conditional covariance given these components.
    Arguments:  comp_expected_cond_cov, numpy.array (num_comps, num_cells, num_cells)
                num_comps_returned, int, 4, return the top n components
    Returns:    sort_order, numpy.array (num_comps)
    """
    num_comps, num_cells, num_cells = comp_expected_cond_cov.shape
    num_combos = num_cells * (num_cells - 1)/2
    ranks = range(num_comps)
    average_rank_by_comp = np.zeros(comp_expected_cond_cov.shape[0], dtype=int)
    for i,j in combinations(range(num_cells), 2):
        average_rank_by_comp[comp_expected_cond_cov[:,i,j].argsort()] += ranks
    average_rank_by_comp = average_rank_by_comp / num_combos
    return np.argsort(average_rank_by_comp)[-num_comps_returned:]

def getEasyJointDistribution(samples, bins):
    """
    For making an empirical joint distribution out of the samples, using the bins.
    Arguments:  samples, each column is a separate variable
                bins, follows the same rules as numpy.histogrammdd's bins argument. (see help(np.histogramdd))
    Returns:    a joint distribution, numpy.array, dimensions controlled by the number of variables, and the bins
    """
    sample_counts, sample_bins = np.histogramdd(samples, bins=bins)
    sample_joint_dist = sample_counts / sample_counts.sum()
    return sample_joint_dist, np.array(sample_bins)

def getCellTopRankCondExp(spike_counts, num_bins_svd, num_comps, top_ranked_value_bins, svd_times, time_bins, top_ranked_joint):
    """
    Helper function for getTopRankConditionalExpectation
    """
    spike_count_list = list(range(spike_counts.min(), spike_counts.max()+1))
    cell_top_ranked_joint_distn = np.zeros(([spike_counts.max()+1] + ([num_bins_svd] * num_comps)), dtype=int) # ex: 29 x 25 x 25 x 25 x 25
    for i,(j,k,l,m) in enumerate(top_ranked_value_bins.T): # looping through all the data that we have is faster than looping through all possible combinations
        svd_bin_value_time = svd_times[i]
        svd_bin_value_time_bin_ind = np.digitize(svd_bin_value_time, time_bins) - 1
        cell_top_ranked_joint_distn[spike_count_list.index(spike_counts[svd_bin_value_time_bin_ind]), j, k, l, m] += 1
    cell_top_ranked_joint_distn = cell_top_ranked_joint_distn / cell_top_ranked_joint_distn.sum()
    with np.errstate(divide='ignore', invalid='ignore'): # avoiding warning messages
        cond_distn = cell_top_ranked_joint_distn / top_ranked_joint
    cond_distn[np.isnan(cond_distn)] = 0.0
    return np.tensordot(np.array(spike_count_list), cond_distn, axes=(0,0))

def getTopRankConditionalExpectation(spike_count_dict, top_ranked_comps, time_bins, svd_times, num_bins_svd):
    """
    For making an empirical joint distributon that includes the top ranked SVD components and the spike counts of a cell.
    Arguments:  spike_count_dict, dict, cell_id => spike_counts
                top_ranked_comps, numpy.array (float) top ranked SVD components,
                top_ranked_bins, numpy.array (float)
                time_bins, numpy.array (float)
                svd_times, numpy.array (float)
                num_bins_svd, int
    Returns:    numpy.array (float)
    """
    num_comps = top_ranked_comps.shape[1]
    num_cells = len(spike_count_dict)
    top_ranked_joint, top_ranked_bins = getEasyJointDistribution(top_ranked_comps, num_bins_svd)
    top_ranked_value_bins = np.array([np.digitize(top_ranked_comps[:,comp_ind], top_ranked_bins[comp_ind])-1 for comp_ind in range(num_comps)])
    top_ranked_value_bins[top_ranked_value_bins == num_bins_svd] = num_bins_svd - 1
    conditional_expectation_dict = {}
    spike_counts_list = list(spike_count_dict.values())
    with Pool() as pool:
        cond_exp_futures = pool.starmap_async(getCellTopRankCondExp, zip(spike_counts_list, num_cells*[num_bins_svd], num_cells*[num_comps], num_cells*[top_ranked_value_bins], num_cells*[svd_times], num_cells*[time_bins], num_cells*[top_ranked_joint]))
        cond_exp_futures.wait()
    conditional_expectation_dict = dict(zip(spike_count_dict.keys(), cond_exp_futures.get()))
    return top_ranked_joint, conditional_expectation_dict

def getAutocorrelation(sequence, num_steps=20):
    return np.array([1]+list(map(lambda x: np.corrcoef(sequence[:-x], sequence[x:])[0,1], range(1,num_steps))))

def getCovarianceOfDictPairs(dictionary):
    """
    For calculating E[XY] in a way that avoids memory issues. Can be used for spike counts and for conditional expectation of spike counts.
    Arguments:  dictionary, cell_id => some array, conditional expectation for each cell, as a function of Z_1,...,Z_M
    Returns:    numpy.array (float) (num_cells, num_cells)
    """
    cell_ids = list(dictionary.keys())
    num_cells = len(cell_ids)
    cov_of_dict_pairs = np.zeros((num_cells, num_cells), dtype=float)
    shuffled_cov = np.zeros((num_cells, num_cells), dtype=float)
    for i,j in combinations(range(num_cells),2):
        cov_of_dict_pairs[i,j] = np.cov(dictionary[cell_ids[i]], dictionary[cell_ids[j]])[0,1]
        shuffled_cov[i,j] = np.cov(dictionary[cell_ids[i]], np.random.permutation(dictionary[cell_ids[j]]))[0,1]
    cov_of_dict_pairs = cov_of_dict_pairs + cov_of_dict_pairs.T
    shuffled_cov = shuffled_cov + shuffled_cov.T
    for i in range(num_cells):
        cell_variance = np.cov(dictionary[cell_ids[i]], dictionary[cell_ids[i]])[0,1]
        cov_of_dict_pairs[i,i] = cell_variance
        shuffled_cov[i,i] = cell_variance
    return cov_of_dict_pairs, shuffled_cov

def downSampleData(svd_times, svd_comps, time_bins, spike_count_dict):
    """
    For converting the svd_comps to a lower frequency sampling. We want to destroy some of the autocorrelation in the PCs.
    Arguments:  svd_times, the times at which the SVD components were measured
                svd_comps, the svd components themselves
                time_bins, the spike count time bin borders
                spike_count_dict, dict, cell_id => spike_counts
    Returns:    down sampled PC times, comps, and spike times and spike count dict that may or may not be downsampled.
    """
    num_counts = time_bins.size - 1
    num_time_points, num_components = svd_comps.shape
    num_time_points_to_skip, remainder = np.divmod(num_time_points, num_counts)
    if num_time_points_to_skip > 0: # downsample PC
        time_bin_svd_times = (np.digitize(time_bins, svd_times) - 1)[1:]
        starting_ind = np.random.randint(0, num_time_points_to_skip)
        time_point_inds = time_bin_svd_times - starting_ind
        new_svd_times, new_svd_comps = svd_times[time_point_inds], svd_comps[time_point_inds,:]
        new_time_bins, new_spike_count_dict = time_bins, spike_count_dict
    elif num_time_points_to_skip == 0: # downsample spike counts
        svd_time_time_bin_inds = np.digitize(svd_times, time_bins)-1
        new_svd_times, new_svd_comps = svd_times, svd_comps
        new_time_bins = time_bins[svd_time_time_bin_inds]
        new_spike_counts = np.array(list(spike_count_dict.values()))[:,svd_time_time_bin_inds]
        new_spike_count_dict = dict(zip(spike_count_dict.keys(), new_spike_counts))
    return new_svd_times, new_svd_comps, new_time_bins, new_spike_count_dict

def fitLinearModel(model, spike_counts, train_inds, test_inds, x_train, x_test):
    """
    For fitting the linear model by cell. Helpful for parallel processing.
    Arguments:  model, the linear model itself, usually an ElasticNetCV,
                spike_counts, the spike counts for the cell,
                train_inds, the training indices,
                test_inds, the test indices,
                x_train, the training features,
                x_test, the test features
    Returns:    r2_score,
                durbin_watson_stat
                fitted_model
    """
    y_train, y_test = spike_counts[train_inds], spike_counts[test_inds]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    model_r2_score = r2_score(y_test, y_pred)
    durbin_watson_stat = durbin_watson(y_test - y_pred)
    return model_r2_score, durbin_watson_stat, model

def getExpCondSpikeCounts(svd_times, svd_comps, time_bins, spike_count_dict):
    """
    For downsampling, fitting, and returning E[X|Z_1,...,Z_m] for every cell X.
    Arguments:  svd_times, numpy array (float), time stamps for the PCs
                svd_comps, numpy array (float), PCs of the mouse face videos
                time_bins, numpy array (float), time stamps for spike counts
                spike_count_dict, dict, cell_id => spike counts
    Returns:    numpy array (float), conditional expectation of the spike counts given the PCs
                linear_model_frame, a dataframe containing information about the linear models, r2 score, durbin-watson score
    """
    num_cells = len(spike_count_dict)
    cell_ids = list(spike_time_dict.keys())
    down_svd_times, down_svd_comps, down_time_bins, down_spike_count_dict = downSampleData(svd_times, svd_comps, time_bins, spike_count_dict) # downsampling to avoid autocorrelation
    num_time_points = down_svd_comps.shape[0]
    train_inds = list(range(num_time_points//2))
    test_inds = list(range(num_time_points//2, num_time_points))
    x_train, x_test = down_svd_comps[train_inds,:], down_svd_comps[test_inds,:]
    alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    elastic_cv_model = ElasticNetCV(l1_ratio=0.5, alphas=alphas, cv=10, n_jobs=-1)
    linear_model_frame = pd.DataFrame(columns=['cell_id', 'r2_score', 'durbin_watson'])
    cond_exp_dict = {}
    for i,(cell_id, spike_counts) in enumerate(down_spike_count_dict.items()):
        model_r2_score, durbin_watson_stat, fitted_model = fitLinearModel(elastic_cv_model, spike_counts, train_inds, test_inds, x_train, x_test)
        linear_model_frame.loc[i] = (cell_id, model_r2_score, durbin_watson_stat)
        cond_exp_dict[cell_id] = fitted_model.predict(down_svd_comps)
    linear_model_frame['num_svd_time_points'] = svd_times.size
    linear_model_frame['num_spike_count_time_points'] = time_bins.size-1
    linear_model_frame['num_time_points_used'] = num_time_points
    linear_model_frame.cell_id = linear_model_frame.cell_id.astype(int)
    return cond_exp_dict, linear_model_frame

def getCondAnalysisFrame(cond_exp_dict, exp_cond_cov, cov_of_cond_expectations, shuff_exp_cond_cov, shuff_cov_of_cond_expectations):
    """
    For creating a conditional analysis frame, analogous to the analysis frame.
    Arguments:  cond_exp_dict, dict => E[X|Z]
                exp_cond_cov, numpy array (num_cells, num_cells), expectation of conditional covariances
                cov_of_cond_expectations, numpy array (num_cells, num_cells), covariance of conditional expectations
    Returns:    pandas Dataframe first_cell_id,second_cell_id,exp_cond_cov,cov_cond_exp,exp_cond_corr,signal_corr
    """
    cond_analysis_frame = pd.DataFrame(columns=['first_cell_id','second_cell_id','exp_cond_cov','cov_cond_exp','exp_cond_corr','signal_corr', 'shuff_exp_cond_cov', 'shuff_cov_cond_exp', 'shuff_exp_cond_corr', 'shuff_signal_corr'])
    cell_ids = list(cond_exp_dict.keys())
    num_cells = len(cell_ids)
    for i,(j,k) in enumerate(combinations(range(num_cells),2)):
        exp_cond_corr = exp_cond_cov[j,k]/np.sqrt(exp_cond_cov[j,j] * exp_cond_cov[k,k]) # this is a funny definition (Maugis 2016 'Event Conditional Correlation')
        shuff_exp_cond_corr = shuff_exp_cond_cov[j,k]/np.sqrt(shuff_exp_cond_cov[j,j] * shuff_exp_cond_cov[k,k])
        signal_corr = cov_of_cond_expectations[j,k]/np.sqrt(np.var(cond_exp_dict[cell_ids[j]]) * np.var(cond_exp_dict[cell_ids[k]]))
        shuff_signal_corr = shuff_cov_of_cond_expectations[j,k]/np.sqrt(np.var(cond_exp_dict[cell_ids[j]]) * np.var(cond_exp_dict[cell_ids[k]]))
        cond_analysis_frame.loc[i] = (cell_ids[j], cell_ids[k], exp_cond_cov[j,k], cov_of_cond_expectations[j,k], exp_cond_corr, signal_corr, shuff_exp_cond_cov[j,k], shuff_cov_of_cond_expectations[j,k], shuff_exp_cond_corr, shuff_signal_corr)
    cond_analysis_frame.first_cell_id = cond_analysis_frame.first_cell_id.astype(int)
    cond_analysis_frame.second_cell_id = cond_analysis_frame.second_cell_id.astype(int)
    return cond_analysis_frame

def getConditionalAnalysisFrame(mouse_face, spike_count_dict, time_bins):
    """
    For getting a conditional analysis frame containing the conditional correlations between spike counts of different cells,
        and getting information on the linear models used to calculate the conditional spike counts.
    Arguments:  mouse_face, dict, contains all info about the mouse films,
                spike_count_dict, Dict, cell_id => spike counts
                time_bins, the spike count time bin borders,
    Returns:    conditional analysis frame, linear model frame
    """
    svd_times, svd_comps = ep.getRelevantMotionSVD(mouse_face, time_bins)
    cond_exp_dict, linear_model_frame = getExpCondSpikeCounts(svd_times, svd_comps, time_bins, spike_count_dict)
    cov_of_spike_counts, shuff_cov_of_spike_counts = getCovarianceOfDictPairs(spike_count_dict) # save this in the conditional analysis frame
    cov_of_cond_expectations, shuff_cov_of_cond_expectations = getCovarianceOfDictPairs(cond_exp_dict) # save this in the conditional analysis frame
    exp_cond_cov = cov_of_spike_counts - cov_of_cond_expectations
    shuff_exp_cond_cov = shuff_cov_of_spike_counts - shuff_cov_of_cond_expectations
    cond_analysis_frame = getCondAnalysisFrame(cond_exp_dict, exp_cond_cov, cov_of_cond_expectations, shuff_exp_cond_cov, shuff_cov_of_cond_expectations)
    return cond_analysis_frame, linear_model_frame, exp_cond_cov, cov_of_cond_expectations

def reduceAnalysisDicts(first_dict, second_dict):
    """
    Each dict is str => numpy array, and will have the same keys. This function appends the dictionaries together. This is the reduce function for mapReduce.
    Arguments:  first_dict, keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
                second_dict, keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
    Return:     Dict, keys=(corr_coef, corr_pv, first_cell_id, plugin_mi, plugin_shuff_mi, second_cell_id, shuff_corr, shuff_corr_pv)
    """
    return {k:np.append(first_dict[k], second_dict[k]) for k in first_dict.keys()}

def saveAnalysisFrame(analysis_frame, chunk_num, save_file):
    """
    Saves the analysis_frame to save_file as a csv with or without a header according to chunk_num.
    If chunk_num == 0, save with header, else append.
    """
    if chunk_num == 0:
        analysis_frame.to_csv(save_file, index=False)
    else:
        analysis_frame.to_csv(save_file, mode='a', header=False, index=False)
    return None

def saveCondFramesMatrices(cond_analysis_frame, linear_model_frame, exp_cond_cov, cov_of_cond_expectations, mouse_name, bin_width):
    """
    For saving the data frames.
    Arguments:  The frames.
    """
    cond_analysis_file = os.path.join(csv_dir, 'conditional_analysis_frames', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + 'conditional_analysis.csv')
    cond_analysis_frame.to_csv(cond_analysis_file, index=False)
    linear_model_frame_file = os.path.join(csv_dir, 'linear_model_frames', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + 'linear_models.csv')
    linear_model_frame.to_csv(linear_model_frame_file, index=False)
    exp_cond_cov_file = os.path.join(npy_dir, 'exp_cond_cov', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + 'exp_cond_cov.npy')
    np.save(exp_cond_cov_file, exp_cond_cov)
    cov_of_cond_expectations_file = os.path.join(npy_dir, 'cov_cond_exp', mouse_name + '_' + str(bin_width).replace('.','p') + '_' + 'cov_cond_exp.npy')
    np.save(cov_of_cond_expectations_file, cov_of_cond_expectations)
    return None

if (not args.debug) & (__name__ == "__main__"):
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Loading cell info...')
    cell_info = pd.read_csv(os.path.join(csv_dir, 'cell_info.csv'), index_col=0)
    for m,mouse_name in enumerate(ep.mouse_names):
        print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing mouse ' + mouse_name + '...')
        mouse_face = ep.loadVideoDataForMouse(mouse_name, mat_dir)
        spon_start_time = ep.spon_start_times[m]
        cell_ids = cell_info[cell_info.mouse_name == mouse_name].index.values
        cell_ids = ep.getRegionallyDistributedCells(cell_info.loc[cell_info.mouse_name == mouse_name], args.number_of_cells)
        spike_time_dict = ep.loadSpikeTimeDict(mouse_name, cell_ids, cell_info, mat_dir)
        pairs = np.array(list(combinations(cell_ids, 2)))
        chunked_pairs = np.array_split(pairs, args.num_chunks)
        for bin_width in ep.selected_bin_widths:
            print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing bin width ' + str(bin_width) + '...')
            spike_count_dict = ep.getSpikeCountDict(spike_time_dict, bin_width, spon_start_time)
            if args.save_firing_rate_frame:
                firing_rate_frame = ep.getFiringRateFrameFromSpikeCountDict(spike_count_dict, bin_width)
                save_file = os.path.join(npy_dir, 'firing_rate_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'firing.npy')
                firing_rate_frame.to_pickle(save_file)
                print(dt.datetime.now().isoformat() + ' INFO: ' + save_file + ' saved.')
            if args.save_analysis_frame:
                save_file = os.path.join(csv_dir, 'analysis_frames', mouse_name + '_' + str(bin_width).replace('.', 'p') + '_' + 'analysis.csv')
                removed = os.remove(save_file) if os.path.exists(save_file) else None
                for i,pair_chunk in enumerate(chunked_pairs):
                    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing chunk number ' + str(i) + '...')
                    analysis_frame = pd.DataFrame.from_dict(futures.mapReduce(getAnalysisDictForPair, reduceAnalysisDicts, constructMapFuncArgs(pair_chunk, spike_count_dict)))
                    analysis_frame['bin_width'] = bin_width
                    saveAnalysisFrame(analysis_frame, i, save_file)
                print(dt.datetime.now().isoformat() + ' INFO: ' + save_file + ' saved.')
            if args.save_conditional_correlations:
                print(dt.datetime.now().isoformat() + ' INFO: ' + 'Processing conditional correlations...')
                cond_analysis_frame, linear_model_frame, exp_cond_cov, cov_of_cond_expectations = getConditionalAnalysisFrame(mouse_face, spike_count_dict, ep.getBinsForSpikeCounts(spike_time_dict, bin_width, spon_start_time))
                saveCondFramesMatrices(cond_analysis_frame, linear_model_frame, exp_cond_cov, cov_of_cond_expectations, mouse_name, bin_width)
                print(dt.datetime.now().isoformat() + ' INFO: Conditional analysis frames and matrices saved.')
            if args.save_spike_count_frame:
                save_file, spike_count_frame = saveSpikeCountFrame(cell_ids, bin_width, spike_time_dict, spon_start_time, mouse_name)
                print(dt.datetime.now().isoformat() + ' INFO: ' + save_file + ' saved.')
    print(dt.datetime.now().isoformat() + ' INFO: ' + 'Done.')
