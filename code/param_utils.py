
def get_wise_params(dataset, emb_txt, emb_vis, direction, model_type):
    '''
    Gets good hyperparameter choices depending on dataset, direction and embeddings (emb_vis & emb_txt)
    '''
    par_learn = {}
    par_learn['dropout'] = 0.0 # set to 0.0 for no dropout
    par_learn['batch_size'] = 128
    par_learn['activation'] = 'relu' # OPTIONS: 'relu' 'sigmoid' 'tanh'

    # ---- DATASET specific choices ---- #

    if dataset == 'imagenet':
        par_learn['n_hidden'] = 512 if emb_vis == 'vgg128' else 1024
        par_learn['lr'] = 0.0001 if direction == 'i2t' else 0.001


    if dataset == 'iaprtc':
        par_learn['n_hidden'] = 512 if emb_vis == 'vgg128' else 1024

        if model_type == 'lin':
            par_learn['lr'] = 0.0001 if direction == 'i2t' else 0.001

        if model_type == 'nn':
            par_learn['lr'] = 0.001


    if dataset == 'wiki':
        par_learn['n_hidden'] = 256 if emb_vis == 'vgg128' else 512

        if direction == 't2i':
            par_learn['lr'] = 0.001

        elif direction == 'i2t':
            if emb_txt == 'biGRU' and emb_vis == 'resnet':
                par_learn['lr'] = 0.0001
            elif emb_txt == 'biGRU' and emb_vis == 'vgg128':
                if model_type == 'lin':
                    par_learn['lr'] = 0.001
                if model_type == 'nn':
                    par_learn['lr'] = 0.0001
            else:
                par_learn['lr'] = 0.001

    return par_learn
