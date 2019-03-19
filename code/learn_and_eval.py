import write_data
import read_data
import param_utils
import eval_utils
import mapping_TF
import data_utils
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='iaprtc', help='OPTIONS: wiki, iaprtc, imagenet')
    parser.add_argument('--n_runs', default=1, type=int, help='Number of runs. (applies to ALL datasets)')
    parser.add_argument('--n_folds', default=5, type=int, help='Number of folds. (applies only to imagenet)')
    # mapping
    parser.add_argument('--epochs_max', default=50, type=int, help='max number of epochs')
    parser.add_argument('--emb_txt', default=['biGRU'],
                        help='List [] of text embeddings. OPTIONS: None (=use all); glove, word2vec (for imagenet); biGRU (wiki & iaprtc)')
    parser.add_argument('--emb_vis', default=['vgg128'], help='List [] of visual embeddings. OPTIONS: None (=use all); vgg128, resnet')
    parser.add_argument('--mappings', default=['lin', 'nn'], help='OPTIONS: lin, nn')
    parser.add_argument('--directions', default=['i2t', 't2i'], help='OPTIONS: i2t (image-to-text), t2i (text-to-image)')
    # performanc measure
    parser.add_argument('--k_neighbors', default=30, type=int, help='number of nearest neighbors to compute the mNNO')
    # directory
    parser.add_argument('--trainDir', default='../training_data/', help='Directory where training data is stored')

    args = parser.parse_args()

    perf_metrics = ['mnno_in-out', 'mnno_map-out', 'mnno_map-in', 'mse', 'R2']
    similarities = ['euclidean', 'cosine'] # (metric to retrieve nearest neighbors in MNNO) OPTIONS: 'euclidean' or 'cosine'

    if args.emb_vis == None:
        args.emb_vis = ['vgg128', 'resnet']
    if args.emb_txt == None:
        args.emb_vis = ['glove', 'word2vec'] if args.dataset == 'imagenet' else ['biGRU']

    saveFolder = write_data.make_save_folder(args.dataset)

    # ---- prevent potential problems ---- #
    if args.dataset == 'imagenet' and len([em for em in args.emb_vis if em in ['resnet', 'vgg128']]) > 0: raise Exception('We do not have such txt embedding in imagenet')
    if args.dataset == 'imagenet' and len([em for em in args.emb_txt if em in ['biGRU',  'word2vec']]) > 0: raise Exception('We do not have such vis embedding in imagenet')
    if args.dataset in ['wiki', 'iaprtc'] and len([em for em in args.emb_txt if em in ['glove', 'word2vec']]) > 0: raise Exception('We do not have such txt embeddings in ' + args.dataset)

    # --- INITIALIZE dictionaries before the loops --- #
    perf_tr, perf_ts, method_compare = {}, {}, []
    for sim in similarities:
        perf_tr[sim], perf_ts[sim] = {}, {}

    # loops:
    for dire in args.directions:
        for em_txt in args.emb_txt:
            for em_vis in args.emb_vis:

                # ---- READ data ---- #
                print('Reading train and test data...')
                X, Y, X_tr, Y_tr, X_ts, Y_ts = read_data.read_and_prepare_dataset(args.dataset, em_txt, em_vis, dire, args.trainDir)
                folds = data_utils.get_folds(X.shape[0], args.n_folds, args.n_runs) if args.dataset == 'imagenet' else [] # folds *only* for Imagenet

                for map in args.mappings:

                    par_learn = param_utils.get_wise_params(args.dataset, em_txt, em_vis, dire, map)

                    method = dire + '_' + em_txt + '_' + em_vis + '_' + map

                    # --- (new method comes in) INITIALIZE performance ditionary of method --- #
                    method_compare.append(method) # include it in the (ordered) list of methods to print
                    print(' ============ METHOD: ', method)
                    for sim in similarities:
                        perf_tr[sim][method], perf_ts[sim][method] = {}, {}
                        for epch in list(range(args.epochs_max)):
                            perf_tr[sim][method][epch], perf_ts[sim][method][epch] = {}, {}
                            for metr in perf_metrics:
                                perf_tr[sim][method][epch][metr], perf_ts[sim][method][epch][metr] = [], []

                    for run_i in range(args.n_runs):

                        if args.dataset == 'imagenet': # we have actual folds for Imagenet only
                            convenient_folds = folds[run_i]
                        elif args.dataset in ['wiki', 'iaprtc']:
                            convenient_folds = [1] # dummy "fold"

                        for j, fold_j in enumerate(convenient_folds):
                            print('******* RUN: ', run_i, ', fold: ', j )

                            if args.dataset == 'imagenet':
                                idx_tr, idx_ts = fold_j['train'], fold_j['test']
                                X_tr, Y_tr, X_ts, Y_ts = X[idx_tr], Y[idx_tr], X[idx_ts], Y[idx_ts]

                            # ------ DEFINE MODEL ------ #
                            model = mapping_TF.Mapping(map, args.dataset, par_learn, X_tr, Y_tr)

                            for epch_k in range(args.epochs_max): # epochs loop
                                print('------> epch: ', epch_k, ' || ', method)

                                # ------ TRAIN in epoch ------ #
                                model.train_epoch(X_tr, Y_tr)

                                # ------ predict EMB_lin and EMB_nn ------ #
                                Y_pred_ts = model.predict(X_ts)
                                Y_pred_tr = model.predict(X_tr)

                                # ------ EVALUATE performance ------ #
                                perf_epch_tr, perf_epch_ts = {}, {}
                                for sim in similarities:
                                    #print('Evaluating performance using: ', sim)
                                    # TODO: calculate mNNO(X,Y) only in the first epoch! (is the same at each epoch)
                                    perf_epch_tr[sim] = eval_utils.eval_performance(X_tr, Y_tr, Y_pred_tr, sim, args.k_neighbors, False)
                                    perf_epch_ts[sim] = eval_utils.eval_performance(X_ts, Y_ts, Y_pred_ts, sim, args.k_neighbors, True)

                                    # --- APPEND results --- #
                                    for metr in perf_metrics:
                                        perf_tr[sim][method][epch_k][metr].append(perf_epch_tr[sim][metr])
                                        perf_ts[sim][method][epch_k][metr].append(perf_epch_ts[sim][metr])

                                    # ------ WRITE results ------ #
                                    write_data.write_results(method_compare, list(range(args.epochs_max)), perf_metrics, perf_tr[sim], saveFolder + '/TR_' + sim + '.csv')
                                    write_data.write_results(method_compare, list(range(args.epochs_max)), perf_metrics, perf_ts[sim], saveFolder + '/TS_' + sim + '.csv')

                                # print only after the second similarity metric
                                print('TEST: MNNO(f(X),X)=', round(perf_epch_ts[sim]['mnno_map-in'],3),
                                      ' || MNNO(f(X),Y)=', round(perf_epch_ts[sim]['mnno_map-out'],3),
                                      ' || MNNO(Y,X)=', round(perf_epch_ts[sim]['mnno_in-out'],3),
                                      ' || mse=', round(perf_epch_ts[sim]['mse'],5), ' || R2=', round(perf_epch_ts[sim]['R2'],4))

                                print('TRAIN: mse=', round(perf_epch_tr[sim]['mse'],5), ' || R2=', round(perf_epch_tr[sim]['R2'],4))

if __name__ == "__main__":
     main()
