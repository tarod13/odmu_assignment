import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=(
    r"\usepackage{amsmath} " 
    + r"\usepackage{amssymb} " 
    + r"\usepackage{mathrsfs}"
))
plt.rc('font', family='serif')
plt.rc('font', size=20)

from src.prophet import ProphetGame
from src.algs import StaticProphetSolver
from src.utils import (
    calculate_average_reward_from_sequences, 
    eliminate_nsamples_repeated
)


def run_evaluation(
    n: int, k: int, 
    n_samples: int,
    means: np.ndarray, 
    vars: np.ndarray, 
    limits: np.ndarray, 
    prophet_samples: int = 1e6,
    verbose: bool = False,
):    
    # Create game
    game = ProphetGame(means, vars, limits, init_prophet=False)
    max_threshold, median_threshold = \
        game.estimate_prophet_value(int(prophet_samples), k)[1:]
    if verbose:
        print(f'Lambda: {max_threshold:.3f}')
        print(f'Eta: {median_threshold:.3f}')

    # Create solvers and set their thresholds 
    max_solver = StaticProphetSolver(n=n, k=k)
    max_solver.set_threshold(max_threshold)

    median_solver = StaticProphetSolver(n=n, k=k)
    median_solver.set_threshold(median_threshold)
    
    # Estimate E[OPT] and E[ALG]
    sequences = game.sample_sequences(n_samples=n_samples)

    max_avg_reward, max_std_reward = \
        calculate_average_reward_from_sequences(max_solver, sequences)
    median_avg_reward, median_std_reward = \
        calculate_average_reward_from_sequences(median_solver, sequences)
    OPT_avg_reward = game.estimate_prophet_value_from_sequences(sequences, k)[0]

    max_competitive_ratio = max_avg_reward / OPT_avg_reward
    median_competitive_ratio = median_avg_reward / OPT_avg_reward

    if verbose:
        print(f'Average reward (OPT): {OPT_avg_reward:.3f}')
        print(
            f'Average reward (lambda): {max_avg_reward:.3f}, ' 
            + f'std: {max_std_reward:.3f}'
        )
        print(
            f'Average reward (eta): {median_avg_reward:.3f}, ' 
            + f'std: {median_std_reward:.3f}'
        )
        print(f'Competitive ratio (lambda): {max_competitive_ratio:.3f}')
        print(f'Competitive ratio (eta): {median_competitive_ratio:.3f}')

    return max_competitive_ratio, median_competitive_ratio


if __name__ == '__main__':

    Path("./plots").mkdir(parents=True, exist_ok=True)

    k = 1
    n = 100
    prophet_samples = 1e5
    n_iters = 50

    # Set game parameters   # TODO: consider other type of distributions
    means = 1+np.ones([n,1]) + np.random.randn(n,1)
    vars = 2*np.random.rand(n,1) + 1e-6
    a = means-2 #np.random.rand(n,)
    b = 20*np.random.rand(n,) + a
    limits = np.stack((a,b), axis=1)
    
    # # Creating outlier
    # means[-1,0] = 100
    # limits[-1,0] = 98
    # limits[-1,1] = 102

    # Execute evaluations
    n_samples_array = np.logspace(
        1, 16.6096405, n_iters, True, 2).astype(np.int64)
    n_samples_array = eliminate_nsamples_repeated(n_samples_array)
    print('Samples considered:')
    print(n_samples_array)
    n_samples_list = n_samples_array.tolist()
    competitive_ratios = {'max':[], 'median':[]}
    for n_samples in tqdm(n_samples_list):
        max_competitive_ratio, median_competitive_ratio = \
            run_evaluation(
                n, k, n_samples, means, vars, limits, prophet_samples, True)

        competitive_ratios['max'].append(max_competitive_ratio)
        competitive_ratios['median'].append(median_competitive_ratio)
    
    for type_, ratios in competitive_ratios.items():
        ratios = np.array(ratios)
    
    fig, ax = plt.subplots(1,1, figsize=(16,10))
    plt.plot(n_samples_array, competitive_ratios['max'], label=r'$\lambda$')
    plt.plot(n_samples_array, competitive_ratios['median'], label=r'$\eta$')
    plt.plot(n_samples_array, 0.5*np.ones_like(n_samples_array), label='Prophet ineq.')
    ax.set_xscale('log')
    ax.set_xlabel('Number of samples', fontsize=20, labelpad=10)
    ax.set_ylabel((
        r'Competitive ratio\hspace{5pt}' 
        + r'$\frac{\mathbb{E}\left[\textsf{ALG}\right]}{\mathbb{E}\left[\textsf{OPT}\right]}$'
        ), 
        fontsize=20, labelpad=10
    )
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.legend()
    plt.grid()
    plt.savefig('./plots/competitive_ratios_k_{k}.pdf', dpi=600)
    plt.show()
    plt.close()
