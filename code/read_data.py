import numpy as np

def readDATA(fileDir):
    words,vectors = [], []
    with open(fileDir) as infile: # wherever you store the vectors
        for line in infile:
            line = line.strip()
            line = line.split(",")
            words.append(line[0])
            vect = line[1:len(line)]
            vect = [float(t) for t in vect]
            vectors.append(vect)
        vectors = np.array(vectors)
    return words, vectors



def read_and_prepare_dataset(dataset, emb_txt, emb_vis, direction, trainDir):
    '''
    Gets X and Y for a dataset
    '''
    X, Y, X_tr, Y_tr, X_ts, Y_ts, = [], [], [], [], [], []

    if dataset == 'imagenet':
        _, EMB_vis = readDATA(trainDir + '/imagenet/vis_' + emb_vis + '_' + emb_txt + '.csv')
        _, EMB_lang = readDATA(trainDir + '/imagenet/txt_' + emb_vis + '_' + emb_txt + '.csv')
        if direction == 't2i':
            X, Y = EMB_lang, EMB_vis
        elif direction == 'i2t':
            X, Y = EMB_vis, EMB_lang

    if dataset in ['wiki', 'iaprtc']:
        lb_ts, EMB_ts_vis = readDATA(trainDir + '/' + dataset + '/TS_vis_' + emb_vis + '.csv')
        _, EMB_ts_lang = readDATA(trainDir + '/' + dataset + '/TS_lang_' + emb_txt + '.csv')
        lb_tr, EMB_tr_vis = readDATA(trainDir + '/' + dataset + '/TR_vis_' + emb_vis + '.csv')
        _, EMB_tr_lang = readDATA(trainDir + '/' + dataset + '/TR_lang_' + emb_txt + '.csv')
        if direction == 't2i':
            X_tr, Y_tr, X_ts, Y_ts = EMB_tr_lang, EMB_tr_vis, EMB_ts_lang, EMB_ts_vis
        elif direction == 'i2t':
            X_tr, Y_tr, X_ts, Y_ts = EMB_tr_vis, EMB_tr_lang, EMB_ts_vis, EMB_ts_lang

    return X, Y, X_tr, Y_tr, X_ts, Y_ts

