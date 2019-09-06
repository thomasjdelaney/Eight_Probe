"""
For analysing the behaviour of the Poisson Binomial distribution. Generates binomial success probabilities using the Beta distribution.
"""
import os, argparse, sys
if float(sys.version[:3]) < 3.0:
    execfile(os.path.join(os.environ['HOME'], '.pystartup'))
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import poibin.poibin as pb
from scipy.stats import beta, poisson

parser = argparse.ArgumentParser(description='A script for experimenting with the Poisson Binomial distribution')
parser.add_argument('-n', '--number_of_binomials', help='Number of binomial variable to sum up.', type=int, default=10)
parser.add_argument('-b', '--beta_params', help='The parameters of the beta distribution', type=float, default=[2.0, 2.0], nargs=2)
parser.add_argument('-s', '--save_figure', help='Indicates that the figure should be saved instead of shown.', default=False, action='store_true')
parser.add_argument('-d', '--debug', help='Enter debug mode.', default=False, action='store_true')
args = parser.parse_args()

proj_dir = os.path.join(os.environ['PROJ'], 'Eight_Probe')
py_dir = os.path.join(proj_dir, 'py')
npy_dir = os.path.join(proj_dir, 'npy')
csv_dir = os.path.join(proj_dir, 'csv')
image_dir = os.path.join(proj_dir, 'images')
mat_dir = os.path.join(proj_dir, 'mat')

beta_dist = beta(args.beta_params[0], args.beta_params[1])
poisson_binomial_params = beta_dist.rvs(size=args.number_of_binomials)
poisson_binomial_dist = pb.PoiBin(poisson_binomial_params)
poisson_possible_outcomes = np.arange(args.number_of_binomials+1)
poisson_binomial_mean = poisson_binomial_params.sum()
poisson_binomial_var = ((1-poisson_binomial_params) * poisson_binomial_params).sum()
poisson_distribution = poisson(poisson_binomial_mean)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(np.arange(0,1.01,0.01), beta_dist.pdf(np.arange(0,1.01,0.01)), label='Beta PDF')
plt.xlabel(r'$p$', fontsize='large'); plt.ylabel(r'$\beta (p)$', fontsize='large');
plt.legend(fontsize='large')

plt.subplot(1,2,2)
plt.plot(poisson_possible_outcomes, poisson_binomial_dist.pmf(poisson_possible_outcomes), label='Poisson-Binomial PMF')
plt.plot(poisson_possible_outcomes, poisson_distribution.pmf(poisson_possible_outcomes), label='Poisson PMF')
plt.xlabel(r'$N$', fontsize='large'); plt.ylabel(r'$PB(N)$', fontsize='large');
plt.title('Mean = ' + str(np.round(poisson_binomial_mean,1)) + ', Variance = ' + str(np.round(poisson_binomial_var,1)), fontsize='large')
plt.legend(fontsize='large')

plt.tight_layout()

file_name = os.path.join(image_dir, 'poisson_binomial_examples', 'poisson_binomial_' + str(args.number_of_binomials) + '_' + str(args.beta_params[0]).replace('.', '-') + '_' + str(args.beta_params[1]).replace('.', '-') + '.png')
plt.savefig(file_name) if args.save_figure else plt.show(block=False)
