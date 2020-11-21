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

from model.resnet import resnet50


class Pass(nn.Module):
	def __init__(self, opt):
		super(Pass, self).__init__()
		self.opt = opt

	def forward(self, feat):

		return feat


class Resnet50(nn.Module):
	def __init__(self, opt):
		super(Resnet50, self).__init__()
		self.resnet = resnet50(pretrained=True)
		self.encoder = nn.Sequential(*list(self.resnet.children())[:-2])


	def forward(self, feat):
		feat = self.encoder(feat)

		return feat


class PointResnet50(nn.Module):
	def __init__(self, opt):
		super(PointResnet50, self).__init__()
		self.encoder = nn.Sequential(*list(self.resnet.children())[:-2])


	def forward(self, feat):
		pdb.set_trace()
		feat = self.encoder(feat)

		return feat
