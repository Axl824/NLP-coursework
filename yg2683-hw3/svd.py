import numpy as np
from collections import Counter
from scipy import sparse
from scipy.sparse.linalg import svds

# import sys

# sys.stdout = open("log2.txt", "w")

# read data
f = 'data/brown.txt'
lines = []
lower_sen = []
for line in open(f, 'r').readlines():
    lines.append(line.strip().split())
for line in lines:
    lower_sen.append([i.lower() for i in line])

# build vocabulary
token_index = dict()
for line in lower_sen:
    for token in line:
        if token not in token_index:
            token_index[token] = len(token_index)
index_token = {j: i for i, j in token_index.items()}
vocab_size = len(token_index)
vocabulary = np.array(list(token_index.keys()))
vocabulary = vocabulary.reshape(-1, 1)


# build co-occurrence matrix
def co_matrix(window):
    co_occur = Counter()
    for line in lower_sen:
        for loc, token in enumerate(line):
            left = max(0, loc - window)
            right = min(len(line) - 1, loc + window)
            contexts = [i for i in range(left, right + 1) if i != loc]
            for context in contexts:
                pair = (token_index[token], token_index[line[context]])
                co_occur[pair] += 1
    rows = []
    cols = []
    vals = []
    for (word, context), count in co_occur.items():
        rows.append(word)
        cols.append(context)
        vals.append(count)
    co_mat = sparse.csr_matrix((vals, (rows, cols)))
    return co_mat


# calculate ppmi
def ppmi_matrix(co_mat):
    total = co_mat.sum()
    partial = co_mat.sum(axis=0).tolist()[0]
    rows, cols = co_mat.nonzero()
    ppmi_mat = co_mat.tolil().copy()
    for i in range(rows.size):
        pmi = np.log((ppmi_mat[rows[i], cols[i]] * total) / (partial[rows[i]] * partial[cols[i]]))
        ppmi_mat[rows[i], cols[i]] = max(0, pmi)
    return ppmi_mat


windows = [2, 5, 10]
dims = [100, 300, 1000]
for window in windows:
    print("Calculate PPMI")
    a = co_matrix(window)
    ppmi = ppmi_matrix(a.astype(float))
    for dim in dims:
        print("Building SVD model\tdimension: %d\twindow size: %d" % (dim, window))
        u, s, vt = svds(ppmi, k=dim)
        embedding = np.multiply(u, np.sqrt(s))
        path = 'svd/svd-' + str(window) + '-' + str(dim) + '.txt'
        with open(path, 'w') as f:
            for i, line in enumerate(embedding):
                out = np.concatenate((vocabulary[i], line))
                f.write(' '.join(map(str, out.tolist()))+'\n')
        print()
