import numpy as np


def average_MSE(y_true, y_pred):
    '''
    :param y_true, y_pred: are the matrices of size (n_samples, n_dim) of predictions and groundtruth resp.
    :return: MSE
    '''
    import numpy as np
    avg_MSE = np.mean(((y_pred - y_true)**2).mean(axis=1))
    return avg_MSE



def eval_performance( EMB_in, EMB_out, EMB_map, similarity, k_neighbors, eval_mnno=True):
    '''
    Computes MNNO, mse and R2 between mapped and input and output embeddings
    '''
    import sklearn.metrics

    assert EMB_in.shape[0] == EMB_out.shape[0] == EMB_map.shape[0], 'Number of samples of the embeddings must match'

    if eval_mnno == True:
        MNNO_in_out, MNNO_map_in, MNNO_map_out = compute_MNNO(EMB_in, EMB_out, EMB_map, similarity, k_neighbors)
    else:
        MNNO_in_out, MNNO_map_in, MNNO_map_out = np.nan, np.nan, np.nan

    mse = average_MSE(EMB_out, EMB_map)

    R2 = sklearn.metrics.r2_score(EMB_out, EMB_map)

    perf_dict = {'mnno_in-out':MNNO_in_out, 'mnno_map-in':MNNO_map_in, 'mnno_map-out':MNNO_map_out, 'mse':mse, 'R2':R2}

    return perf_dict



def compute_MNNO(EMB_in, EMB_out, EMB_map, similarity='cosine', k_neighbors=10):
    '''
    *IMPORTANT* Samples *must* be paired across all embedding types (in, out and map).
    :param EMB_*: either the input, output, or mapped samples each of shape (n_samples, n_dim).
    :param similarity: 'cosine' or 'euclidean' to find neighbors
    :param k_neighbors: number of neighbors to consider
    :return: dictionary of MNNOs
    '''
    from sklearn.neighbors import NearestNeighbors

    neighs_map = NearestNeighbors(n_neighbors=(k_neighbors), metric=similarity, algorithm='brute').fit(EMB_map)
    distances_map, indices_map = neighs_map.kneighbors(EMB_map)
    neighs_in = NearestNeighbors(n_neighbors=(k_neighbors), metric=similarity, algorithm='brute').fit(EMB_in)
    distances_in, indices_in = neighs_in.kneighbors(EMB_in)
    neighs_out = NearestNeighbors(n_neighbors=(k_neighbors), metric=similarity, algorithm='brute').fit(EMB_out)
    distances_out, indices_out = neighs_out.kneighbors(EMB_out)

    # Compute nearest neighbor overlap for each sample
    overlap_in_map = []
    overlap_out_map = []
    overlap_in_out = []
    for sampl in range(EMB_in.shape[0]): # loop over words in input embedding (it is the smallest subset)
        try:
            # get nearest neighbors of sampl for each embedding type
            neighs_word_map = indices_map[sampl, 0:]
            neighs_word_in = indices_in[sampl, 0:]
            neighs_word_out = indices_out[sampl, 0:]
            # compute overlap between found neighbors:
            overlap_in_map.append(len([id for id in neighs_word_map if id in neighs_word_in]))
            overlap_out_map.append(len([id for id in neighs_word_map if id in neighs_word_out]))
            overlap_in_out.append(len([id for id in neighs_word_in if id in neighs_word_out]))
        except:
            print('Sample ', sampl, '  could not be computed')

    # average (and normalize by number of neighbors)
    MNNO_in_out = np.mean(overlap_in_out)/float(k_neighbors)
    MNNO_map_in = np.mean(overlap_in_map)/float(k_neighbors)
    MNNO_map_out = np.mean(overlap_out_map)/float(k_neighbors)

    return MNNO_in_out, MNNO_map_in, MNNO_map_out

