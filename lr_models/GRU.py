import math
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        self.gru.flatten_parameters()
        h0 = x.new_zeros(self.num_layers*2, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        feat = out # self.dropout(out)
        out = self.dropout(out)
        fc = self.fc(out).mean(1)
        sf = fc.softmax(-1)
        # return (self.fc(out).log_softmax(-1), out)
        # return (self.fc(out), out)
        return (sf, fc, feat)

def gen_GRU():
    # model = GRU(512, 1024, 2, 500)
    model = GRU(512, 1024, 2, 500)
    # GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses)
    return model
    
def gen_GRU_fab():
    # model = GRU(512, 1024, 2, 500)
    model = GRU(256, 1024, 2, 500)
    # GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses)
    return model