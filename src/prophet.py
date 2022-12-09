import numpy as np
from scipy.stats import truncnorm
from tqdm import tqdm

from src.utils import find_k_max


class ProphetGame:
    def __init__(
        self, 
        means: np.ndarray, 
        vars: np.ndarray, 
        limits: np.ndarray, 
        rng: np.random.Generator = None,
        n_samples_prophet: int = 10000,
        init_prophet: bool = True,
        **kwargs
    ):
        '''Create prophet game with truncated Gaussian distributions.'''
        self.steps = means.shape[0]
        self.means = means
        self.stds = vars**0.5
        self.limits = limits
        self.standard_limits = (limits - means) / self.stds
        self.rng = rng    
        if init_prophet:
            self.estimate_sub_prophet_values(n_samples_prophet)
   
    def sample_sequences(self, n_samples) -> np.ndarray:
        '''Sample the given number of sequences.'''
        a = self.standard_limits[:,0].reshape([-1,1])
        b = self.standard_limits[:,1].reshape([-1,1])
        samples = truncnorm.rvs(
            a=a, b=b, 
            loc=self.means, scale=self.stds,
            size=(a.shape[0], n_samples), random_state=self.rng
        )
        return samples
    
    def estimate_prophet_value(self, n_samples: int = 10000, k: int = 1):
        # Sample the given number of sequences
        sequences = self.sample_sequences(n_samples)

        # Estimate prophet value and thresholds
        expected_cumulative_reward, max_threshold, median_threshold = \
            self.estimate_prophet_value_from_sequences(sequences, k)
        
        return expected_cumulative_reward, max_threshold, median_threshold

    def estimate_prophet_value_from_sequences(
        self, sequences: np.ndarray, k: int
    ):
        # Select the k max values
        max_k_values = find_k_max(sequences, k)

        # Sum the top k values
        cumulative_reward = max_k_values.sum(0)

        # Average across all samples
        expected_cumulative_reward = cumulative_reward.mean()
        
        # Set lambda as 0.5 times OPT
        max_threshold = 0.5*expected_cumulative_reward
        
        # Calculate the median to set eta
        median_threshold = np.median(cumulative_reward)
        return expected_cumulative_reward, max_threshold, median_threshold

    def estimate_sub_prophet_values(self, n_samples: int = 10000):
        '''Calculate prophet values for all possible subsequences.'''
        self.prophet_dict = {
            'prophet_values' : {}, 
            'max_thresholds': {},
            'median_thresholds': {},
        }
        vars = self.stds**2
        for l in tqdm(range(1, self.steps+1)):
            for i in range(0, self.steps-l+1):
                # Set values for sub-game
                sub_means = self.means[i:i+l,:]
                sub_vars = vars[i:i+l,:]
                sub_limits = self.limits[i:i+l,:]
                
                # Create sub-game
                sub_game = ProphetGame(
                    sub_means, sub_vars, sub_limits, init_prophet=False)

                # Estimate prophet value and thresholds for sub-game
                sub_prophet_value, sub_max_threshold, sub_median_threshold = (
                    sub_game.estimate_prophet_value(n_samples, k=1))

                # Store estimation (l,i).
                # l is the lenght of the subsequence, 
                # and i its starting position.
                self.prophet_dict['prophet_values'][(l,i)] = (
                    sub_prophet_value)
                self.prophet_dict['max_thresholds'][(l,i)] = (
                    sub_max_threshold)
                self.prophet_dict['median_thresholds'][(l,i)] = (
                    sub_median_threshold)


if __name__ == "__main__":

    k = 2
    n_samples = 10

    means = np.ones([10,1])
    vars = np.ones([10,1])
    a = -1*np.ones(10,)
    b = 3*np.ones(10,)
    limits = np.stack((a,b), axis=1)

    game = ProphetGame(means, vars, limits)
    pv, max_t, median_t = game.estimate_prophet_value(n_samples, k)
    print(f'Prophet value: {pv:.3f}')
    print(f'Lambda: {max_t:.3f}')
    print(f'Eta: {median_t:.3f}')
    print('Prophet dictionary:')
    print(game.prophet_dict)
