import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
from torchvision.utils import make_grid
import time
from torch.utils.data import Dataset

import torch.nn.functional as F

class Net(nn.Module):
    def init(self):
        super(Net,self).init()
        
        self.fres = 29
        self.predensen=36
        #Convolutions
        self.conv1 = nn.Conv2d(3,9,kernel_size=5) 
        self.conv2 = nn.Conv2d(9,18,kernel_size=5)
        self.conv3 = nn.Conv2d(18,36, kernel_size =3)
        self.conv4 = nn.Conv2d(36,36, kernel_size = 3)
        self.pool = nn.MaxPool2d(2,2)
        self.dense = nn.Linear(self.predensen*self.fres*self.fres,14)
        self.dense2= nn.Linear(81,14)
        
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1,self.predensen*self.fres*self.fres)
        x = self.dense(x)
        return nn.Sigmoid(x)