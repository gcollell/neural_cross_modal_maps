
def get_folds(n_samples, n_folds=5, n_runs=2):
    '''
    Get folds for imagenet dataset
    '''
    import numpy as np
    folds = {}
    for run_i in range(n_runs):
        indices = np.random.permutation(np.arange(n_samples))
        n_test = int(np.floor(n_samples/ n_folds))
        folds[run_i] = [{'test':indices[i*n_test:(i+1)*n_test], 'train':np.delete(indices, np.arange(i*n_test,(i+1)*n_test))} for i in range(n_folds)]
    return folds

