import numpy as np

def writeCSV(words, matrix, saveDir):
    # words: a 1-D numpy array or a list
    # matrix: numpy array
    import csv
    matrix = np.array(matrix)
    #1. build the "joint" matrix
    MATRIX = []
    for i in range(len(words)):
        vec = matrix[i,]
        currentWord = [ words[i] ]
        currentWord.extend(vec)
        MATRIX.append(currentWord)
    # 2. write the CSV file
    with open(saveDir, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(MATRIX)


def write_results(method_compare, epochs_vec, perf_metrics, perf_dic, saveDir):
    '''
    perf_dic is a list of dictionaries (starting at epoch 0)
    For each epoch we will have (possibly) many entries (one per each run and/or fold)
    '''
    measures = ['method', 'epch'] + perf_metrics
    MATRIX = []
    MATRIX.append(measures)
    for method in method_compare: # perf_dic[method][epch]
        for epch in epochs_vec:
            new_row = [method, epch] + [ np.mean(perf_dic[method][epch][metr]) for metr in perf_metrics ]
            MATRIX.append(new_row)
    MATRIX = np.array(MATRIX)
    matrix2csv(MATRIX, saveDir)


def matrix2csv(MATRIX, saveDir):
    #MATRIX is an np.array
    import csv as csv
    with open(saveDir, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(MATRIX)


def make_save_folder(dataset):
    import os
    save_name = '../results/' + dataset
    if os.path.exists(save_name):
        for idx_dir in range(50): # 50 is just a limit for number of generated files
            saveFolder = save_name + '(' + str(idx_dir) + ')'
            if not os.path.exists(saveFolder):
                break
    else:
        saveFolder = save_name
    os.makedirs(saveFolder)
    return saveFolder