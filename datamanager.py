import numpy as np
from utils import *

class datamanager_mnist(object):
    def __init__(self, train_ratio=None, fold_k=None, norm=False, expand_dim=False, seed=233):
        self.seed = seed
        mnist = np.load("mnist.npz") # https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
        self.data = np.concatenate([mnist["x_train"], mnist["x_test"]])
        self.labels = np.concatenate([mnist["y_train"], mnist["y_test"]])
        self.labels = one_hot_encode(self.labels, 10)
        del mnist
        
        self.divide_train_test(train_ratio, fold_k)

        if norm:
            self.data = self.data / 255. # [0,1]
            # self.data = self.data / 127.5 - 1. # [-1, 1]
        
        self.train_cur_pos, self.test_cur_pos = 0, 0

        self.expand_dim = expand_dim

    def get_train_test_id(self, train_ratio, fold_k, seed=None):
        self.train_id, self.test_id = [], []
        # normal
        if train_ratio and not fold_k:
            for v in self.dict_by_class:
                self.train_id += v[:int(train_ratio * len(v))]
                self.test_id += v[int(train_ratio * len(v)):]
        # cross validation
        if fold_k and not train_ratio:
            for i in range(10):
                self.test_id += list(self.dict_by_class[i][self.test_fold_id])
                for j in range(fold_k):
                    if j != self.test_fold_id:
                        self.train_id += list(self.dict_by_class[i][j])
        self.train_id = np.array(self.train_id)
        self.test_id = np.array(self.test_id)

        shuffle_in_unison_scary(self.train_id, self.test_id, seed=(seed or self.seed))
        self.train_num, self.test_num = len(self.train_id), len(self.test_id)

    def divide_train_test(self, train_ratio, fold_k, seed=None):
        self.dict_by_class = [[] for i in range(10)]

        for i, key in enumerate(np.argmax(self.labels, axis=1)):
            self.dict_by_class[key].append(i)
        for i in range(10):
            np.random.seed(i)
            np.random.shuffle(self.dict_by_class[i])
        
        if fold_k and not train_ratio:
            # only for cross validation
            print ("[{} folds cross validation]".format(fold_k))
            for i in range(10):
                np.random.seed(i)
                np.random.shuffle(self.dict_by_class[i])
                self.dict_by_class[i] = np.array_split(self.dict_by_class[i], fold_k)
                np.random.seed(i)
                np.random.shuffle(self.dict_by_class[i])
            self.test_fold_id = 0
        self.get_train_test_id(train_ratio, fold_k, seed)
    
    def shuffle_train(self, seed=None):
        np.random.seed(seed)
        np.random.shuffle(self.train_id)
    
    def get_cur_pos(self, cur_pos, full_num, batch_size):
        get_pos = range(cur_pos, cur_pos + batch_size)
        if cur_pos + batch_size <= full_num:
            cur_pos += batch_size
        else:
            rest = cur_pos + batch_size - full_num
            get_pos = list(range(cur_pos, full_num)) + list(range(rest)) # range(cur_pos, full_num) + range(rest)
            cur_pos = rest
        return cur_pos, get_pos

    def __call__(self, batch_size, phase='train', maxlen=None, var_list=[]):
        if phase == 'train':
            self.train_cur_pos, get_pos = self.get_cur_pos(self.train_cur_pos, self.train_num, batch_size)
            cur_id = self.train_id[get_pos]
        elif phase == 'test':
            self.test_cur_pos, get_pos = self.get_cur_pos(self.test_cur_pos, self.test_num, batch_size)
            cur_id = self.test_id[get_pos]

        def func(flag, maxlen=maxlen):
            if flag == 'data':
                res = self.__dict__[flag][cur_id]
                if self.expand_dim:
                    res = np.expand_dims(res, -1)
                return res
            elif flag == 'labels':
                return self.labels[cur_id]
        
        res = {}
        for key in var_list:
            if key not in res:
                res[key] = func(key)
        
        return res

# data = datamanager_mnist(1.0, None, True, True)
# tmp = []
# for _ in range(5):
#     x = data(100, var_list=['data'])
#     tmp.append(x['data'])
# p,_ = np.histogram(tmp, bins=10, density=False)
# print p
# print np.sum(p)
