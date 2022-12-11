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
    prophet_value = game.estimate_prophet_value(int(prophet_samples), k)[0]
    
    # Create solver and set threshold 
    ALG_solver = StaticProphetSolver(n=n, k=k)
    ALG_solver.set_prophet_threshold(prophet_value)

    # Estimate E[OPT] and E[ALG]
    sequences = game.sample_sequences(n_samples=n_samples)

    ALG_avg_reward, ALG_std_reward = \
        calculate_average_reward_from_sequences(ALG_solver, sequences)
    OPT_avg_reward = game.estimate_prophet_value_from_sequences(sequences, k)[0]

    competitive_ratio = ALG_avg_reward / OPT_avg_reward
    
    if verbose:
        print(f'Average reward (OPT): {OPT_avg_reward:.3f}')
        print(f'Average reward (ALG): {ALG_avg_reward:.3f}, ' 
            + f'std: {ALG_std_reward:.3f}'
        )
        print(f'Competitive ratio: {competitive_ratio:.3f}')
        
    return competitive_ratio


if __name__ == '__main__':

    Path("./plots").mkdir(parents=True, exist_ok=True)

    n = 100
    prophet_samples = 1e5
    n_samples = 100000
    k_min = 1
    k_max = n

    # Set game parameters   # TODO: consider other type of distributions
    means = 0*np.linspace(3,6,n).reshape(-1,1) + 1*np.ones([n,1]) + 1*np.random.randn(n,1)
    vars = 2*np.random.rand(n,1) + 1e-6
    a = np.random.rand(n,)
    b = 0*np.ones_like(a) + 5*np.random.rand(n,) + a
    limits = np.stack((a,b), axis=1)

    # Execute evaluations
    k_list = list(range(k_min,k_max+1))
    k_array = np.array(k_list)
    competitive_ratios = []
    prophet_constants = 1/(k_array+1)
    for k in tqdm(k_list):
        competitive_ratio = run_evaluation(
          n, k, n_samples, means, vars, limits, prophet_samples, True)
        competitive_ratios.append(competitive_ratio)
        
    competitive_ratios = np.array(competitive_ratios)
    
    fig, ax = plt.subplots(1,1, figsize=(16,10))
    plt.scatter(k_array, competitive_ratios, label=f'Competitive ratios')
    plt.scatter(k_array, prophet_constants, label=f'Prophet ineq.')
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
    plt.ylim((0,1))
    plt.savefig(f'./plots/competitive_ratios_general.pdf', dpi=600)
    plt.show()
    plt.close()