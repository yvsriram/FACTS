import numpy as np
from scipy.stats import kendalltau

def get_pairwise_similarity(order1, order2, measure='kendalltau'):
    if measure == 'kendalltau':
        return kendalltau(order1, order2)[0]
    elif measure == 'pearson':
        return np.corrcoef(order1, order2)[0][1]
    else:
        raise NotImplementedError