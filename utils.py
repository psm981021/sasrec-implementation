import sys
import copy
import torch
import random
import numpy as np
import pickle as pkl
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(f'data/Beauty/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    
    return [user_train, user_valid, user_test, usernum, itemnum]

def get_data_dic(args):
    dat = pkl.load(open(f'{args.data_dir}{args.data_name}_all_multi_word.dat', 'rb'))
    data = {}

    user_reviews = dat['user_seq_token']
    data['user_seq_wt'] = []
    data['user_seq'] = []
    for u in user_reviews:
        data['user_seq_wt'].append(user_reviews[u])
        items = [item for item, time in user_reviews[u]]
        data['user_seq'].append(items)

    data['user_seq_wt_dic'] = user_reviews
    data['items_feat'] = dat['items_feat']
    data['n_items'] = len(dat['item2id'])
    data['n_users'] = len(dat['user2id']) - 1
    data['n_categories'] = len(dat['category2id'])
    data['n_brands'] = len(dat['brand2id'])
    data['feature_size'] = 6 + 1 + data['n_categories'] + data['n_brands'] - 2
    data['sample_seq'] = get_user_sample(args.data_dir + args.data_name + '_sample.txt')
    return data


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# DLFS 코드 추가

def get_user_sample(sample_file):
    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)
    return sample_seq


def get_data_dic(args):
    dat = pkl.load(open(f'{args.data_dir}{args.dataset}_all_multi_word.dat', 'rb'))
    data = {}

    user_reviews = dat['user_seq_token']
    data['user_seq_wt'] = []
    data['user_seq'] = []
    for u in user_reviews:
        data['user_seq_wt'].append(user_reviews[u])
        items = [item for item, time in user_reviews[u]]
        data['user_seq'].append(items)

    data['user_seq_wt_dic'] = user_reviews
    data['items_feat'] = dat['items_feat']
    data['n_items'] = len(dat['item2id'])
    data['n_users'] = len(dat['user2id']) - 1
    data['n_categories'] = len(dat['category2id'])
    data['n_brands'] = len(dat['brand2id'])
    data['feature_size'] = 6 + 1 + data['n_categories'] + data['n_brands'] - 2
    data['sample_seq'] = get_user_sample(args.data_dir + args.dataset + '_sample.txt')
    return data

def get_feats_vec(feats, args):
    feats = torch.tensor(feats)
    feat_category = torch.zeros(feats.size(0), args['n_categories'])
    category_vec = feat_category.scatter_(index=feats[:, 1:-1].long(), value=1, dim=-1)
    feat_brand = torch.zeros(feats.size(0), args['n_brands'])
    brand_vec = feat_brand.scatter_(index=feats[:, -1:].long(), value=1, dim=-1)
    vec = torch.cat((feats[:, :1], category_vec[:, 1:], brand_vec[:, 1:]), dim=1)
    return vec