import numpy as np
import torch
import random

__all__ = ['Sampler', 'DataSampler']


class Sampler(object):
    """Sampler base class.
    A sampler is meant to be used as a generator of batches useful in training neural networks.
    Notes
    -----
    Each new sampler must extend this base class implementing all the abstract
    special methods.
    """

    def __init__(self, *args, **kargs):
        pass

    def __len__(self):
        """Return the number of batches.
        """
        raise NotImplementedError

    def __iter__(self):
        """Iterate through the batches yielding a batch at a time.
        """
        raise NotImplementedError


class DataSampler(Sampler):
    def __init__(self,
                 data,
                 user_item_matrix,
                 i_u_matrix,
                 n_items,
                 n_neg_samples=1,
                 batch_size=128,
                 shuffle=True,
                 seed=2022,
                 device='cuda:0'):
        super(DataSampler, self).__init__()
        self.data = data
        self.user_item_matrix = user_item_matrix
        self.i_u_matrix = i_u_matrix
        self.n_items = n_items
        self.n_neg_samples = n_neg_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __len__(self):
        dataset_size = 0
        length = self.data.groupby("history_length")  # histories could be of different lengths, so we need to group
        for i, (_, l) in enumerate(length):
            dataset_size += int(np.ceil(l.shape[0] / self.batch_size))
        return dataset_size

    def __iter__(self):
        length = self.data.groupby(["history_length", "type"])  # histories could be of different lengths, so we need to group
        for i, (_, l) in enumerate(length):
            group_users = np.array(list(l['userID']))
            group_items = np.array(list(l['itemID']))
            group_rating = np.array(list(l['rating']))
            group_histories = np.array(list(l['history']))
            group_feedbacks = np.array(list(l['history_feedback']))
            group_type = np.array(list(l['type']))

            n = group_users.shape[0]
            idxlist = list(range(n))
            if self.shuffle:  # every small dataset based on history length is shuffled before preparing batches
                np.random.shuffle(idxlist)

            for _, start_idx in enumerate(range(0, n, self.batch_size)):
                end_idx = min(start_idx + self.batch_size, n)
                batch_users = torch.from_numpy(group_users[idxlist[start_idx:end_idx]])
                batch_items = torch.from_numpy(group_items[idxlist[start_idx:end_idx]])
                batch_rating = torch.from_numpy(group_rating[idxlist[start_idx:end_idx]])
                batch_histories = torch.from_numpy(group_histories[idxlist[start_idx:end_idx]])
                batch_feedbacks = torch.from_numpy(group_feedbacks[idxlist[start_idx:end_idx]])
                batch_type = torch.from_numpy(group_type[idxlist[start_idx:end_idx]])

                # here, we generate negative items for each interaction in the batch
                batch_user_item_matrix = self.user_item_matrix[batch_users].toarray()  # this is the portion of the
                # user-item matrix for the users in the batch
                batch_user_unseen_items = 1 - batch_user_item_matrix  # this matrix contains the items that each user
                # in the batch has never seen
                negative_items = []  # this list contains a list of negative items for each interaction in the batch
                for u in range(batch_users.size(0)):
                    u_unseen_items = batch_user_unseen_items[u].nonzero()[0]  # items never seen by the user
                    # here, we generate n_neg_samples indexes and use them to take n_neg_samples random items from
                    # the list of the items that the user has never seen
                    rnd_negatives = u_unseen_items[random.sample(range(u_unseen_items.shape[0]), self.n_neg_samples)]
                    # we append the list to negative_items
                    negative_items.append(rnd_negatives)

                # 给每个item找一百个neg_user
                batch_i_u_matrix = self.i_u_matrix[batch_items].toarray()
                batch_i_unseen_u = 1 - batch_i_u_matrix
                negative_users = []
                for i in range(batch_items.size(0)):
                    i_unseen_users = batch_i_unseen_u[i].nonzero()[0]
                    rnd_negatives = i_unseen_users[random.sample(range(i_unseen_users.shape[0]), self.n_neg_samples)]
                    negative_users.append(rnd_negatives)

                batch_negative_items = torch.tensor(np.array(negative_items))
                batch_negative_users = torch.tensor(np.array(negative_users))
                batch_negative_users += self.n_items
                batch_users += self.n_items
                yield batch_users.to(self.device), batch_items.to(self.device), batch_rating.to(self.device), batch_histories.to(self.device), \
                      batch_feedbacks.to(self.device), batch_negative_items.to(self.device), batch_type.to(self.device), \
                      batch_negative_users.to(self.device)
