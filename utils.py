
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm


def purge_ppl(array, indices):
    """ Reduce number of people label for balancing """
    print(len(indices))
    droprate_ppl = 0.5
    droprate_empty = 0.9
    valid_indices = []
    for i in range(len(indices)):
        if array[indices[i]][9] == 1:
            if np.random.rand() > droprate_ppl:
                valid_indices = valid_indices + [indices[i]]
        elif np.array_equal(array[indices[i]], [0 for i in range(14)]):
            if np.random.rand() > droprate_empty:
                valid_indices = valid_indices + [indices[i]]
        else:
            valid_indices = valid_indices + [indices[i]]

    print(len(valid_indices))
    return valid_indices


def capsamples(capsize, array, indices):
    """ All label balacing """
    total_per_label = [0 for i in range(14)]
    newindices = []
    tagged = []
    empty = 0
    for i in range(len(indices)):
        if np.array_equal(array[indices[i]], [0 for i in range(14)]):
            if empty < capsize:
                newindices = newindices + [i]
                empty += 1

        else:
            aux = array[indices[i]] + total_per_label
            flag = True
            for j in aux:
                if j > capsize:
                    flag = False
            if flag == True:
                newindices = newindices + [i]
                total_per_label = aux

    print(total_per_label)
    return newindices

class Kfolder:
    """ Cross validation """

    def __init__(self, dataset, folds_number):
        self.k = folds_number
        self.dataset = dataset

        self.fold_size = 1 / folds_number
        # Generate FOLDS
        lengths = [
            int((self.fold_size) * len(self.dataset)) for i in range(0, folds_number)
        ]
        self.folds = torch.utils.data.random_split(dataset, lengths)

        # Apply transformations to every fold
        for i in range(len(self.folds)):
            self.folds[i].transform = dataset.transform

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < self.k:
            validation = self.folds[self.n]
            training = []
            for i in range(len(self.folds)):
                if i != self.n:
                    training = training + [self.folds[i]]

            self.n += 1
            return [training, validation]
        else:
            raise StopIteration

# F-score, precision and recall
def coincidences(array1, array2):
    TP = 0.0
    FP = 0.0
    TN = 0.0
    FN = 0.0
    # treshold en los tensores
    t = nn.Threshold(0.5, 0)
    a1 = t(array1)

    for i in range(len(a1)):
        # print(a1[i])
        # print(array2[i])
        for j in range(len(a1[i])):
            if a1[i][j] > 0 and array2[i][j] > 0:
                TP += 1.0
            elif a1[i][j] > 0 and array2[i][j] < 0.5:
                FP += 1.0
            elif a1[i][j] < 0.5 and array2[i][j] > 0:
                FN += 1.0
            else:
                TN += 1.0
        # print("TP: %d, FP: %d, TN: %d, FN: %d" % (TP,FP,TN,FN))
    return TP, FP, TN, FN
