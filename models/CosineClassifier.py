import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


class CosineClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.weight = Parameter(torch.Tensor(n_classes, input_dim))
        self.eta = Parameter(torch.Tensor(1))
        self.init_weights()
        
    def init_weights(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.eta.data.fill_(1)    

    def forward(self, x):
        x_norm = F.normalize(x, p=2,dim=1)
        w_norm = F.normalize(self.weight, p=2,dim=1)
        y = self.eta * F.linear(x_norm, w_norm)
        return y
    