import os
import pdb
import math
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from pytorchcv.model_provider import get_model as ptcv_get_model
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FC(nn.Module):
    def __init__(self, opt):
        super(FC, self).__init__()
        self.cls = nn.Sequential(
        	nn.Linear(2048, 512),
        	nn.ReLU(),
        	nn.Linear(512, 4)
    	)

    def forward(self, feat):
        feat = self.cls(feat)

        return feat


class PointFC(nn.Module):
    def __init__(self, opt):
        super(PointFC, self).__init__()
        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

        self.reg = nn.Linear(256, 2)

    def forward(self, feat):
        chord = self.cls(feat.mean(1))
        finger = self.reg(feat)

        return chord, finger