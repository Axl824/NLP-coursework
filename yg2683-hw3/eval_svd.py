from process import load_model
from evaluate import collect, eval_wordsim, eval_bats, eval_msr


windows = [2, 5, 10]
dims = [100, 300, 1000]

for window in windows:
    for dim in dims:
        print("SVD model\tdimension: %d\twindow size: %d" % (dim, window))
        path = 'svd/svd-' + str(window) + '-' + str(dim) + '.txt'

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
