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

_, valset, testset = utils.get_dataset(opt)
valloader = DataLoader(valset, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.num_workers)
testloader = DataLoader(testset, batch_size=opt.batch_size,
                        shuffle=False, num_workers=opt.num_workers)

encode, middle, decode = encode.eval(), middle.eval(), decode.eval()
with torch.no_grad():
    res, gt = [], []
    for step, pack in enumerate(valloader):
        img, y, hd, imgp = pack
        if opt.gpu:
            img = img.cuda()

        out = encode(img)
        out = middle(out)
        out_chord, out_finger = decode(out)
        out_chord, out_finger = out_chord.cpu(), out_finger.cpu()

        res.append(out_chord)
        gt.append(y)

    res = torch.cat(res, 0)
    gt = torch.cat(gt, 0)
    top1, top2, top3 = utils.accuracy(res, gt, (1,)), utils.accuracy(res, gt, (2,)), utils.accuracy(res, gt, (3,))
    print('val acc is {:.4f}, {:.4f}, {:.4f}'.format(top1[0].item(), top2[0].item(), top3[0].item()))

with torch.no_grad():
    res, gt = [], []
    for step, pack in enumerate(testloader):
        img, y, hd, imgp = pack
        if opt.gpu:
            img = img.cuda()

        out = encode(img)
        out = middle(out)
        out_chord, out_finger = decode(out)
        out_chord, out_finger = out_chord.cpu(), out_finger.cpu()

        res.append(out_chord)
        gt.append(y)

    res = torch.cat(res, 0)
    gt = torch.cat(gt, 0)
    top1, top2, top3 = utils.accuracy(res, gt, (1,)), utils.accuracy(res, gt, (2,)), utils.accuracy(res, gt, (3,))
    print('test acc is {:.4f}, {:.4f}, {:.4f}'.format(top1[0].item(), top2[0].item(), top3[0].item()))