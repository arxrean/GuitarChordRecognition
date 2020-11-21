import os
import cv2
import pdb
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from option import get_parser
import utils

opt = get_parser()

encode, middle, decode = utils.get_model(opt)
best_pth = torch.load(os.path.join(
    'save', opt.name, 'check', 'best.pth.tar'), map_location='cpu')
encode.load_state_dict(best_pth['encode'])
middle.load_state_dict(best_pth['middle'])
decode.load_state_dict(best_pth['decode'])
if opt.gpu:
    encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()

_, _, testset = utils.get_dataset(opt)
testloader = DataLoader(testset, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.num_workers)

encode, middle, decode = encode.eval(), middle.eval(), decode.eval()

prototype_set = testset.prototype
with torch.no_grad():
    out = encode(prototype_set['prototype'])
    out = middle(out)
    prototype_feat = out.reshape(out.size(0), -1)

with torch.no_grad():
    res, gt = [], []
    for step, pack in enumerate(testloader):
        img, y, hd, imgp = pack
        if opt.gpu:
            img = img.cuda()

        out = encode(img)
        out = middle(out)

        res.append(out)
        gt.append(y)

    res = torch.cat(res, 0)
    res = res.reshape(res.size(0), -1)
    gt = torch.cat(gt, 0).numpy()

scores = F.softmax(torch.mm(res, prototype_feat.transpose(0, 1)), -1).numpy()
scores = np.argsort(scores, -1)

seen_scores = scores[gt!=4]
unseen_scores = scores[gt==4]
seen_gt = gt[gt!=4]
unseen_gt = gt[gt==4]

print('top-1 seen: {}'.format(np.mean([1 if s[-1]==g else 0 for s, g in zip(seen_scores, seen_gt)])))
print('top-2 seen: {}'.format(np.mean([1 if g in s[-2:] else 0 for s, g in zip(seen_scores, seen_gt)])))
print('top-1 unseen: {}'.format(np.mean([1 if s[-1]==g else 0 for s, g in zip(unseen_scores, unseen_gt)])))
print('top-2 unseen: {}'.format(np.mean([1 if g in s[-2:] else 0 for s, g in zip(unseen_scores, unseen_gt)])))


