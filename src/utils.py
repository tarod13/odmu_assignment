import numpy as np


def find_k_max(x, k):
    '''Returns the largest k values in each column of the matrix x.'''
    if k > x.shape[0]:
        # raise ValueError(
        #     f'k can be at most the lenght of the sequences. ' + 
        #     f'Got k={k} and lenght={x.shape[0]}')
        k = x.shape[0]

    max_indices = np.argpartition(x, kth=-k, axis=0)[-k:]
    max_values = x[max_indices, np.arange(x.shape[1])]
    return max_values


def find_length_combinations(n: int, k: int):
    '''
    Finds all possible combinations of k lenghts that 
    sum to n, where each lenght is at least 1.
    '''
    if min(k,n) == 1:
        return [[n]]
    elif min(k,n) > 1:
        lengths = []
        for i in range(1, n-(k-1)+1):
            future_lengths = find_length_combinations(n-i, k-1)
            for fl in future_lengths:
                lengths.append([i]+fl)
        return lengths
    else:
        raise ValueError(
            "Invalid k or n. They must be larger or equal to 1.")

def length_combination2id_sequence(combination):
    ids = []
    for l in combination:
        if len(ids) == 0:
            ids.append(l)
        else:
            ids.append(l + ids[-1])
    return ids 

def calculate_average_reward_from_sequences(solver, sequences):
    selected_elements = solver.select_at_most_k(sequences)
    cumulative_reward = np.stack(selected_elements, axis=0).sum(0)
    avg_cumulative_reward = cumulative_reward.mean()
    std_cumulative_reward = cumulative_reward.std()
    return avg_cumulative_reward, std_cumulative_reward

def eliminate_nsamples_repeated(array_):
    for i in range(1,len(array_)):
        if array_[i] <= array_[i-1]:
            array_[i] = array_[i-1]+1
    return array_