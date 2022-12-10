import numpy as np
from abc import ABC, abstractmethod

from src.prophet import ProphetGame
from src.utils import find_length_combinations, length_combination2id_sequence


class ProphetSolver(ABC):
    def __init__(self, n: int, k: int = 1):
        self.n = n
        self.k = k
        self.thresholds = k*[0]
        self.threshold_ids = k*[0]   # TODO: take these values into account (?)
    
    @abstractmethod
    def set_threshold(self, prophet_values):
        ...
        
    def select_at_most_k(self, sequences):
        n_samples = sequences.shape[1]
        valid_sequences = sequences.copy()
        selected_elements = []
        for i in range(self.k):
            # Creates fiter for values that overcome threshold 
            over_threshold = valid_sequences > self.thresholds[i]
            
            # Determine samples for which at least 
            # one value overcomes threshold 
            valid_columns = over_threshold.max(axis=0, keepdims=True)
            
            # Filter out samples with no good values, as per the threhold
            valid_sequences *= valid_columns

            # Find the first element in each sample that overcomes the threshold
            indices_i_element = np.argmax(over_threshold, axis=0)

            # Store selected elements for each sample 
            selected_elements.append(
                valid_sequences[indices_i_element, np.arange(n_samples)].copy()
            )

            # Remove selected elements from the samples
            valid_sequences[indices_i_element, np.arange(n_samples)] = 0
        return selected_elements        

class IndependentProphetSolver(ProphetSolver):
    def set_threshold(self, prophet_dict: dict):
        length_combinations = find_length_combinations(self.n, self.k)
        best_combination = None
        best_thresholds = None
        max_value = 0.0
        for combination in length_combinations:
            # Calculate value associated to each combination of lengths
            value = 0.0
            thresholds = []
            for i, l in enumerate(combination):
                value += prophet_dict['prophet_values'][(l,i)]
                thresholds.append(prophet_dict['max_thresholds'][(l,i)])

            # Update best combination if found better value 
            if (best_combination is None) or (value > max_value):
                best_combination = combination
                best_thresholds = thresholds
                max_value = value                

        self.thresholds = best_thresholds
        self.threshold_ids = length_combination2id_sequence(best_combination)


class StaticProphetSolver(ProphetSolver):
    def set_threshold(self, threshold: float):
        self.thresholds = self.k*[threshold]

    def set_prophet_threshold(self, prophet_value: float):
        self.set_threshold(prophet_value/(self.k + 1))


if __name__ == '__main__':

    n = 10
    k = 2
    n_samples = 100000

    # Set game parameters
    means = np.ones([n,1]) + np.random.randn(n,1)
    vars = np.ones([n,1])
    a = -0*np.ones(n,)
    b = 3*np.ones(n,)
    limits = np.stack((a,b), axis=1)

    # Create game
    game = ProphetGame(means, vars, limits)
    pv = game.estimate_prophet_value(100000, k)[0]
    print(f'Prophet value: {pv:.3f}')

    # Create solver and set its thresholds based on 
    solver = IndependentProphetSolver(n=n, k=k)
    solver.set_thresholds(game.prophet_dict)
    print(f'Thresholds: {solver.thresholds}')
    print(f'Threshold_ids: {solver.threshold_ids}')

    sequences = game.sample_sequences(n_samples=n_samples).round(2)
    selected_elements = solver.select_at_most_k(sequences)
    cumulative_reward = np.stack(selected_elements, axis=0).sum(0)
    expected_cumulative_reward = cumulative_reward.mean()
    print(f'Cumulative reward obtained: {expected_cumulative_reward:.3f}')
    print(f'Competitive ratio: {expected_cumulative_reward / pv:.3f}')
