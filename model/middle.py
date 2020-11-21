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

from model.gcn import GAT


class AvgPool(nn.Module):
	def __init__(self, opt):
		super(AvgPool, self).__init__()

	def forward(self, feat):
		feat = feat.mean(-1).mean(-1)

		return feat


class Point(nn.Module):
	def __init__(self, opt):
		super(Point, self).__init__()
		self.index = nn.Parameter(torch.randn(2048))
		self.middle = nn.Parameter(torch.randn(2048))
		self.ring = nn.Parameter(torch.randn(2048))
		self.pinky = nn.Parameter(torch.randn(2048))

		self.finger_encode = nn.Linear(2048, 128)
		self.gat = GAT(opt)
		self.adj = nn.Parameter(torch.ones(4, 4))

	def forward(self, feat):
		index = self.attention_finger_feat(feat, self.index)
		middle = self.attention_finger_feat(feat, self.middle)
		ring = self.attention_finger_feat(feat, self.ring)
		pinky = self.attention_finger_feat(feat, self.pinky)

		finger_feat = torch.stack([index, middle, ring, pinky]).transpose(0, 1)
		adj = torch.bmm(finger_feat, finger_feat.transpose(1, 2))
		out = torch.stack([self.gat(x, adj[i]) for i, x in enumerate(finger_feat)])

		return out

	def attention_finger_feat(self, feat, finger):
		feat = feat.reshape(feat.size(0), feat.size(1), -1)
		out = torch.bmm(finger.unsqueeze(-1).unsqueeze(0).repeat(feat.size(0), 1, 1).transpose(1, 2), feat)
		out = F.softmax(out, -1)
		out = torch.sum(feat * out, -1)

		out = self.finger_encode(out)

		return out