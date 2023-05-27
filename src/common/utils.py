import numpy as np


# np_similarity_score = lambda a,b:sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))

def np_seq_similarity_score(seq_a, seq_b):
    s = (seq_a/np.linalg.norm(seq_a, axis=-1, keepdims=True))@(seq_b/np.linalg.norm(seq_b, axis=-1, keepdims=True)).T
    return np.mean(s)