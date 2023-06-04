import argparse
from model.data import Dataset
from model.samplers import DataSampler
from model.nets import Cbox4CR
from model.models import Cbox4CRTrainer
from model.evaluation import ValidFunc, logic_evaluate
import torch
import pandas as pd
import logging
import os

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
os.environ['NUMEXPR_MAX_THREADS'] = '20'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set pytorch device for computation

def main():
    # init args
    init_parser = argparse.ArgumentParser(description='Experiment')
    init_parser.add_argument('--threshold', type=int, default=4,
                             help='Threshold for generating positive/negative feedback.')
    init_parser.add_argument('--order', type=bool, default=True,
                             help='Flag indicating whether the ratings have to be ordered by timestamp on not.')
    init_parser.add_argument('--leave_n', type=int, default=1,
                             help='Number of positive interactions that are hold out from each user for validation and test sets.')
    init_parser.add_argument('--keep_n', type=int, default=5,
                             help='Minimum number of positive interactions that are kept in training set for each user.')
    init_parser.add_argument('--max_history_length', type=int, default=5,
                             help='Maximum length of history for each interaction (i.e. maximum number of items at the left of the implication in the logical expressions)')
    init_parser.add_argument('--n_neg_train', type=int, default=1,
                             help='Number of negative items randomly sampled for each training interaction. The items are sampled from the set of items that the user has never seen.')
    init_parser.add_argument('--n_neg_val_test', type=int, default=100,
                             help='Number of negative items randomly sampled for each validation/test interaction. The items are sampled from the set of items that the user has never seen.')
    init_parser.add_argument('--training_batch_size', type=int, default=128,
                             help='Size of training set batches.')
    init_parser.add_argument('--val_test_batch_size', type=int, default=256,
                             help='Size of validation/test set batches.')
    init_parser.add_argument('--seed', type=int, default=2022,
                             help='Random seed to reproduce the experiments.')
    init_parser.add_argument('--emb_size', type=int, default=64,
                             help='Size of users, item, and event embeddings.')
    init_parser.add_argument('--lr', type=float, default=0.0005,
                             help='Learning rate for the training of the model.')
    init_parser.add_argument('--l2', type=float, default=0.0001,
                             help='Weight for the L2 regularization.')
    init_parser.add_argument('--val_metric', type=str, default='ndcg@5',
                             help='Metric computed for the validation of the model.')
    init_parser.add_argument('--test_metrics', type=list, default=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'],
                             help='Metrics computed for the test of the model.')
    init_parser.add_argument('--n_epochs', type=int, default=100,
                             help='Number of epochs for the training of the model.')
    init_parser.add_argument('--early_stop', type=int, default=20,
                             help='Number of epochs for early stopping. It should be > 1.')
    init_parser.add_argument('--at_least', type=int, default=20,
                             help='Minimum number of epochs before starting with early stopping.')
    init_parser.add_argument('--save_load_path', type=str, default="saved-models/best_model.json",
                             help='Path where the model has to be saved during training. The model is saved every time the validation metric increases. This path is also used for loading the best model before the test evaluation.')
    init_parser.add_argument('--n_times', type=int, default=10,
                             help='Number of times the test evaluation is performed (metrics are averaged across these n_times evaluations). This is required since the negative items are randomly sampled.')
    init_parser.add_argument('--dataset', type=str, default="movielens_100k",
                             help="Dataset on which the experiment has to be performed ('movielens_1m', 'amazon_movies_tv', 'amazon_electronics').")
    init_parser.add_argument('--test_only', type=bool, default=False,
                             help='Flag indicating whether it has to be computed only the test evaluation or not. If True, there should be a model checkpoint to load in the specified save path.')
    # contrastive learning parameters
    init_parser.add_argument('--position_encoder', type=bool, default=True,
                             help='add rope for nets and augmentation.')
    init_parser.add_argument('--tau', type=float, default=0.01,
                             help='temperature coefficient.')
    init_parser.add_argument('--theta', type=float, default=0.02,
                             help='recommendation loss weight.')
    init_parser.add_argument('--beta', type=float, default=0.4,
                             help='augmentation coefficient.')
    init_parser.add_argument('--activation', type=str, default='relu',
                             help="activation function in transform('none','relu','softplus')")
    # box recommendation loss parameters
    init_parser.add_argument('--cen', type=float, default=0.02)
    init_parser.add_argument('--gamma', type=float, default=12.0)
    init_parser.add_argument('--epsilon', type=float, default=2.0)

    # other parameter
    init_parser.add_argument('--num_layers', type=int, default=1,
                             help="number of mlp's layers.")
    init_parser.add_argument('--std_vol', type=float, default=-2,
                             help="used to balance box loss.")
    init_parser.add_argument('--u2i', type=bool, default=True,
                             help="use u2i nodes to train.")



    init_args, init_extras = init_parser.parse_known_args()
    # take the correct dataset
    if init_args.dataset == "movielens_100k":
        raw_dataset = pd.read_csv("datasets/movielens-100k/movielens_100k.csv")
    if init_args.dataset == "amazon_movies_tv":
        raw_dataset = pd.read_csv("datasets/amazon-movies-tv/movies_tv.csv")
    if init_args.dataset == "amazon_electronics":
        raw_dataset = pd.read_csv("datasets/amazon-electronics/electronics.csv")
    if init_args.dataset == "ml-1m":
        raw_dataset = pd.read_csv("datasets/ml-1m/ml-1m.csv")
    if init_args.dataset == "ml-10m":
        raw_dataset = pd.read_csv("datasets/ml-10m/ml-10m.csv")

    # create train, validation, and test sets
    dataset = Dataset(raw_dataset, init_args.u2i)
    dataset.process_data(threshold=init_args.threshold, order=init_args.order, leave_n=init_args.leave_n,
                         keep_n=init_args.keep_n, max_history_length=init_args.max_history_length)
    if not init_args.test_only:
        train_loader = DataSampler(dataset.train_set, dataset.user_item_matrix, dataset.i_u_matrix, dataset.n_items,
                                   n_neg_samples=init_args.n_neg_train,
                                   batch_size=init_args.training_batch_size, shuffle=True, seed=init_args.seed,
                                   device=device)
        val_loader = DataSampler(dataset.validation_set, dataset.user_item_matrix, dataset.i_u_matrix, dataset.n_items,
                                 n_neg_samples=init_args.n_neg_val_test,
                                 batch_size=init_args.val_test_batch_size,
                                 shuffle=False, seed=init_args.seed, device=device)
    test_loader = DataSampler(dataset.test_set, dataset.user_item_matrix, dataset.i_u_matrix, dataset.n_items,
                              n_neg_samples=init_args.n_neg_val_test,
                              batch_size=init_args.val_test_batch_size, shuffle=False, seed=init_args.seed,
                              device=device)
    cbox4cr_nets = Cbox4CR(dataset.n_users, dataset.n_items, init_args.num_layers,
                           init_args.gamma, init_args.epsilon, init_args.cen, init_args.activation,
                           init_args.beta, init_args.tau, init_args.std_vol,
                           emb_size=init_args.emb_size, seed=init_args.seed).to(device)

    cbox4cr_model = Cbox4CRTrainer(cbox4cr_nets, learning_rate=init_args.lr, l2_weight=init_args.l2,
                           theta=init_args.theta)

    if not init_args.test_only:
        cbox4cr_model.train(train_loader, valid_data=val_loader, valid_metric=init_args.val_metric,
                        valid_func=ValidFunc(logic_evaluate), num_epochs=init_args.n_epochs,
                        at_least=init_args.at_least, early_stop=init_args.early_stop,
                        save_path=init_args.save_load_path, verbose=1)

    cbox4cr_model.load_model(init_args.save_load_path)

    cbox4cr_model.test(test_loader, test_metrics=init_args.test_metrics, n_times=init_args.n_times)


if __name__ == '__main__':
    main()
