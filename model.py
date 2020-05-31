
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
import torchvision
from skimage import io, transform
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets.utils import download_url
from torchvision.utils import make_grid
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fres = 30
        self.predensen = 27
        # Convolutions
        self.conv1 = nn.Conv2d(3, 9, kernel_size=3)
        self.conv2 = nn.Conv2d(9, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 27, kernel_size=3)
        self.conv4 = nn.Conv2d(27, 27, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dense = nn.Linear(self.predensen * self.fres * self.fres, 14)
        self.dense2 = nn.Linear(81, 14)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        # x = self.pool(torch.relu(self.conv4(x)))
        # print(x.shape)
        x = x.view(-1, self.predensen * self.fres * self.fres)
        x = self.dense(x)
        # x = self.dense2(x)
        return x