import numpy as np
from abc import ABC, abstractmethod

from secretary import SecretaryGame
from utils import find_length_combinations, length_combination2id_sequence


class SecretarySolver(ABC):
    def __init__(self, n: int, k: int = 1):
        self.n = n
        self.k = k
        self.thresholds = k*[0]
        self.threshold_ids = k*[0]
    
    @abstractmethod
    def set_thresholds(self, prophet_value: float):
        ...
        
    def select_at_most_k(self, sequences):
        n_samples = sequences.shape[1]
        valid_sequences = sequences.copy()
        selected_elements = []
        for i in range(self.k): 
            over_threshold = valid_sequences > self.thresholds[i]
            valid_columns = over_threshold.max(axis=0, keepdims=True)
            valid_sequences *= valid_columns
            indices_i_element = np.argmax(over_threshold, axis=0)
            selected_elements.append(
                valid_sequences[indices_i_element, np.arange(n_samples)].copy()
            )
            valid_sequences[indices_i_element, np.arange(n_samples)] = 0
        return selected_elements        

class IndependentSecretarySolver(SecretarySolver):
    def set_thresholds(self, prophet_dict: float):
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


if __name__ == '__main__':

    n = 10
    k = 2
    n_samples = 100000

    means = np.ones([n,1]) + np.random.randn(n,1)
    vars = np.ones([n,1])
    a = -0*np.ones(n,)
    b = 3*np.ones(n,)
    limits = np.stack((a,b), axis=1)

    game = SecretaryGame(means, vars, limits)
    pv = game.estimate_prophet_value(100000, k)[0]
    print(f'Prophet value: {pv:.3f}')

    solver = IndependentSecretarySolver(n=n, k=k)
    solver.set_thresholds(game.prophet_dict)
    print(f'Thresholds: {solver.thresholds}')
    print(f'Threshold_ids: {solver.threshold_ids}')

    sequences = game.sample_sequences(n_samples=n_samples).round(2)
    selected_elements = solver.select_at_most_k(sequences)
    cumulative_reward = np.stack(selected_elements, axis=0).sum(0)
    expected_cumulative_reward = cumulative_reward.mean()
    print(f'Cumulative reward obtained: {expected_cumulative_reward:.3f}')
    print(f'Competitive ratio: {expected_cumulative_reward / pv:.3f}')