import itertools as it
from gensim.models import word2vec
from process import load_model
from evaluate import collect, eval_wordsim, eval_bats, eval_msr

# import sys

# sys.stdout = open("log.txt", "w")

f = 'data/brown.txt'
lines = []
lower_sen = []
for line in open(f, 'r').readlines():
    lines.append(line.strip().split())
for line in lines:
    lower_sen.append([i.lower() for i in line])

windows = [2, 5, 10]
dims = [100, 300, 1000]
negsamples = [1, 5, 15]
parameters = [windows, dims, negsamples]
combs = list(it.product(*parameters))

for window, dim, neg in combs:
    path = 'w2v/W2V-' + str(window) + '-' + str(dim) + '-' + str(neg) + '.bin'
    print("Word2vec model\tdimension: %d\twindow size: %d\tnegative samples: %d" % (dim, window, neg))
    
    model = word2vec.Word2Vec(sentences=lower_sen, size=dim, window=window, sg=1, negative=neg)
    model.wv.save_word2vec_format(path, binary=True)
    model = load_model(path)

    print('[evaluate] Collecting matrix...')
    matrix, vocab, indices = collect(model)

    print('[evaluate] WordSim353 correlation:')
    ws = eval_wordsim(model)
    print(ws)

    print('[evaluate] BATS accuracies:')
    bats = eval_bats(model, matrix, vocab, indices)
    print(bats)

    print('[evaluate] MSR accuracy:')
    msr = eval_msr(model)
    print(msr)
    print()
