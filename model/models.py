import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .evaluation import ValidFunc, logic_evaluate
from .net_utils import bpr_loss

__all__ = ['Cbox4CRTrainer']

logger = logging.getLogger(__name__)


class Cbox4CRTrainer(object):

    def __init__(self, net, learning_rate=0.001, l2_weight=1e-4,alpha=1.0, theta=0.02):
        self.network = net
        self.lr = learning_rate
        self.l2_weight = l2_weight
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        self.alpha = alpha
        self.theta = theta

    def loss_function(self, positive_logit, negative_logit, c_loss):
        rec_loss = bpr_loss(positive_logit, negative_logit, self.theta)
        return rec_loss+c_loss

    def train(self,
              train_data,
              valid_data=None,
              valid_metric=None,
              valid_func=ValidFunc(logic_evaluate),
              num_epochs=100,
              at_least=20,
              early_stop=5,
              save_path="../saved_models/best_model.json",
              verbose=1):

        best_val = 0.0
        early_stop_counter = 0
        early_stop_flag = False
        if early_stop > 1:  # it means that the parameter is meaningful
            early_stop_flag = True
        try:
            for epoch in range(1, num_epochs + 1):
                self.train_epoch(epoch, train_data, verbose)
                if valid_data is not None:
                    assert valid_metric is not None, \
                        "In case of validation 'valid_metric' must be provided"
                    valid_res = valid_func(self, valid_data, valid_metric)
                    mu_val = np.mean(valid_res)
                    std_err_val = np.std(valid_res) / np.sqrt(len(valid_res))
                    logger.info('| epoch %d | %s %.4f (%.4f) |',
                                epoch, valid_metric, mu_val, std_err_val)
                    if mu_val > best_val:
                        best_val = mu_val
                        self.save_model(save_path, epoch)  # save model if an improved validation score has been
                        # obtained
                        early_stop_counter = 0  # we have reached a new validation best value, so we put the early stop
                        # counter to zero
                    else:
                        # if we did not have obtained an improved validation metric, we have to increase the early
                        # stopping counter
                        if epoch >= at_least and early_stop_flag:  # we have to train for at least 20 epochs, they said that in the paper
                            early_stop_counter += 1
                            if early_stop_counter == early_stop:
                                logger.info('Traing stopped at epoch %d due to early stopping', epoch)
                                break
        except KeyboardInterrupt:
            logger.warning('Handled KeyboardInterrupt: exiting from training early')

    def train_epoch(self, epoch, train_loader, verbose=1):

        self.network.train()  # set the network in train mode
        train_loss = 0
        partial_loss = 0
        epoch_start_time = time.time()
        start_time = time.time()
        log_delay = max(10, len(train_loader) // 10 ** verbose)

        for batch_idx, batch_data in enumerate(train_loader):
            partial_loss += self.train_batch(batch_data)
            if (batch_idx + 1) % log_delay == 0:
                elapsed = time.time() - start_time
                logger.info('| epoch %d | %d/%d batches | ms/batch %.2f | loss %.2f |',
                            epoch, (batch_idx + 1), len(train_loader),
                            elapsed * 1000 / log_delay,
                            partial_loss / log_delay)
                train_loss += partial_loss
                partial_loss = 0.0
                start_time = time.time()
        total_loss = (train_loss + partial_loss) / len(train_loader)
        time_diff = time.time() - epoch_start_time
        logger.info("| epoch %d | loss %.4f | total time: %.2fs |", epoch, total_loss, time_diff)

    def train_batch(self, batch_data):

        self.optimizer.zero_grad()
        positive_preds, negative_preds, c_loss = self.network(batch_data)
        loss = self.loss_function(positive_preds, negative_preds, c_loss)
        loss.backward()
        # this gradient clipping leads to lower results, so I removed it
        # torch.nn.utils.clip_grad_value_(self.network.parameters(), 50)  # this has been inserted in the code provided
        self.optimizer.step()
        return loss.item()

    def predict(self, batch_data):
        self.network.eval()  # we have to set the network in evaluation mode
        with torch.no_grad():
            positive_predictions, negative_predictions, _ = self.network(batch_data)
        return positive_predictions, negative_predictions

    def save_model(self, filepath, cur_epoch):

        logger.info("Saving model checkpoint to %s...", filepath)
        torch.save({'epoch': cur_epoch,
                    'state_dict': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, filepath)
        logger.info("Model checkpoint saved!")

    def load_model(self, filepath):

        assert os.path.isfile(filepath), "The checkpoint file %s does not exist." % filepath
        logger.info("Loading model checkpoint from %s...", filepath)
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self.network.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Model checkpoint loaded!")
        return checkpoint

    def test(self, test_loader, test_metrics=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'], n_times=10):

        metric_dict = {}
        for i in range(n_times):  # compute test metrics n_times times and take the mean since negative samples are
            # randomly generated
            evaluation_dict = logic_evaluate(self, test_loader, test_metrics)
            for metric in evaluation_dict:
                if metric not in metric_dict:
                    metric_dict[metric] = {}
                metric_mean = np.mean(evaluation_dict[metric])
                metric_std_err_val = np.std(evaluation_dict[metric]) / np.sqrt(len(evaluation_dict[metric]))
                if "mean" not in metric_dict[metric]:
                    metric_dict[metric]["mean"] = metric_mean
                    metric_dict[metric]["std"] = metric_std_err_val
                else:
                    metric_dict[metric]["mean"] += metric_mean
                    metric_dict[metric]["std"] += metric_std_err_val

        for metric in metric_dict:
            logger.info('%s: %.4f (%.4f)', metric, metric_dict[metric]["mean"] / n_times,
                        metric_dict[metric]["std"] / n_times)
