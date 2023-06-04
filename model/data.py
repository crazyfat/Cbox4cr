import pandas as pd
from scipy import sparse
import copy
pd.set_option('display.max_columns', None)

__all__ = ['Dataset']


class Dataset(object):

    def __init__(self, raw_dataset, u2i):
        self.dataset = raw_dataset
        self.use_u2i = u2i
        self.n_users = self.dataset['userID'].nunique()
        self.n_items = self.dataset['itemID'].nunique()
        self.user_item_matrix = self.compute_sparse_matrix()
        self.i_u_matrix = self.compute_i_u_matrix()

    def compute_sparse_matrix(self):
        group = self.dataset.groupby("userID")
        rows, cols = [], []
        values = []
        for i, (_, g) in enumerate(group):
            u = list(g['userID'])[0]  # user id
            items = set(list(g['itemID']))  # items on the history
            rows.extend([u] * len(items))
            cols.extend(list(items))
            values.extend([1] * len(items))
        return sparse.csr_matrix((values, (rows, cols)), (self.n_users, self.n_items))

    def compute_i_u_matrix(self):
        group = self.dataset.groupby("itemID")
        rows, cols = [], []
        values = []
        for i, (_, g) in enumerate(group):
            i = list(g['itemID'])[0]
            users = set(list(g['userID']))
            rows.extend([i] * len(users))
            cols.extend(list(users))
            values.extend([1] * len(users))
        return sparse.csr_matrix((values, (rows, cols)), (self.n_items, self.n_users))

    def process_data(self, threshold=4, order=True, leave_n=1, keep_n=5, max_history_length=5, premise_threshold=0):
        # filter ratings by threshold
        self.proc_dataset = self.dataset.copy()
        print(self.proc_dataset)
        self.proc_dataset['rating'][self.proc_dataset['rating'] < threshold] = 0
        self.proc_dataset['rating'][self.proc_dataset['rating'] >= threshold] = 1
        if order:
            self.proc_dataset = self.proc_dataset.sort_values(by=['timestamp', 'userID', 'itemID']).reset_index(
                drop=True)

        self.leave_one_out_by_time(leave_n, keep_n)
        self.generate_histories(max_hist_length=max_history_length, premise_threshold=0)

    def leave_one_out_by_time(self, leave_n=1, keep_n=5):

        train_set = []
        # generate training set by looking for the first keep_n POSITIVE interactions
        processed_data = self.proc_dataset.copy()
        for uid, group in processed_data.groupby('userID'):  # group by uid
            found, found_idx = 0, -1
            for idx in group.index:
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= keep_n:
                        break
            if found_idx > 0:
                train_set.append(group.loc[:found_idx])
        train_set = pd.concat(train_set)
        # drop the training data info
        processed_data = processed_data.drop(train_set.index)

        # generate test set by looking for the last leave_n POSITIVE interactions
        test_set = []
        for uid, group in processed_data.groupby('userID'):
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            if found_idx > 0:
                test_set.append(group.loc[found_idx:])
        test_set = pd.concat(test_set)
        processed_data = processed_data.drop(test_set.index)

        validation_set = []
        for uid, group in processed_data.groupby('userID'):
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            # put all the negative interactions encountered during the search process into validation set
            if found_idx > 0:
                validation_set.append(group.loc[found_idx:])
        validation_set = pd.concat(validation_set)
        processed_data = processed_data.drop(validation_set.index)

        # The remaining data (after removing validation and test) are all in training data
        self.train_set = pd.concat([train_set, processed_data])
        self.validation_set, self.test_set = validation_set.reset_index(drop=True), test_set.reset_index(drop=True)

    def generate_histories(self, max_hist_length=5, premise_threshold=0):
        # TODO insert an assert to check the value of the parameter premise_threshold (read doc)
        history_dict = {}  # it contains for each user the list of all the items he has seen
        feedback_dict = {}  # it contains for each user the list of feedbacks he gave to the items he has seen
        for df in [self.train_set, self.validation_set, self.test_set]:
            history = []  # each element of this list is a list containing the history items of a single interaction
            fb = []  # each element of this list is a list containing the feedback for the history items of a
            # single interaction
            hist_len = []  # each element of this list indicates the number of history items of a single interaction
            type = []  # 2表示u2i节点， 1表示i2u
            uids, iids, feedbacks = df['userID'].tolist(), df['itemID'].tolist(), df['rating'].tolist()
            for i, uid in enumerate(uids):
                iid, feedback = iids[i], feedbacks[i]

                if uid not in history_dict:
                    history_dict[uid] = []
                    feedback_dict[uid] = []

                # list containing the history for current interaction
                tmp_his = copy.deepcopy(history_dict[uid]) if max_hist_length == 0 else history_dict[uid][
                                                                                        -max_hist_length:]
                # list containing the feedbacks for the history of current interaction
                fb_his = copy.deepcopy(feedback_dict[uid]) if max_hist_length == 0 else feedback_dict[uid][
                                                                                        -max_hist_length:]

                history.append(tmp_his)
                fb.append(fb_his)
                hist_len.append(len(tmp_his))
                if tmp_his == []:
                    type.append(2)
                else:
                    type.append(4)
                history_dict[uid].append(iid)
                feedback_dict[uid].append(feedback)

            df['history'] = history
            df['history_feedback'] = fb
            df['history_length'] = hist_len
            df['type'] = type

        if premise_threshold != 0:
            self.train_set = self.train_set[self.train_set.history_length > premise_threshold]
            self.validation_set = self.validation_set[self.validation_set.history_length > premise_threshold]
            self.test_set = self.test_set[self.test_set.history_length > premise_threshold]

        self.clean_data()

    def clean_data(self):
        train_type1 = self.train_set.copy()
        train_type1['type'] = 1
        type1 = train_type1.sample(frac=1.0, replace=False, random_state=2022, axis=0)
        self.train_set = self.train_set[self.train_set['rating'] > 0].reset_index(drop=True)
        self.train_set = self.train_set[self.train_set['history_feedback'].map(len) >= 1].reset_index(drop=True)
        self.validation_set = self.validation_set[self.validation_set['rating'] > 0].reset_index(drop=True)
        self.validation_set = self.validation_set[self.validation_set['history_feedback'].map(len) >= 1].reset_index(
            drop=True)
        self.test_set = self.test_set[self.test_set['rating'] > 0].reset_index(drop=True)
        self.test_set = self.test_set[self.test_set['history_feedback'].map(len) >= 1].reset_index(drop=True)
        if self.use_u2i:
            self.train_set = pd.concat([type1, self.train_set], axis=0, ignore_index=True)
