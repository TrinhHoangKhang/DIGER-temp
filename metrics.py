import numpy as np


def dcg(scores):
    scores = np.asarray(scores, dtype=np.float64)                                       
    return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))

def ndcg_at_k(r, k):
    r = np.asarray(r, dtype=np.float64)[:k]                                                        
    dcg_max = dcg(sorted(r, reverse=True))
    if not dcg_max:
        return 0.
    return dcg(r) / dcg_max
