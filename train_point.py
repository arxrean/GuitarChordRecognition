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

utils.init_log_dir(opt)
writer = SummaryWriter('./save/{}/tb'.format(opt.name))

encode, middle, decode = utils.get_model(opt)
if opt.gpu:
	encode, middle, decode = encode.cuda(), middle.cuda(), decode.cuda()

loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam([{'params': encode.parameters()},
						{'params': middle.parameters()},
						{'params': decode.parameters()}], opt.base_lr, weight_decay=opt.weight_decay)

trainset, valset, _ = utils.get_dataset(opt)
trainloader = DataLoader(trainset, batch_size=opt.batch_size,
						 shuffle=True, num_workers=opt.num_workers, drop_last=True)
valloader = DataLoader(valset, batch_size=opt.batch_size,
					   shuffle=False, num_workers=opt.num_workers)

best_val_acc = 0.
for epoch in range(opt.epoches):
	encode, middle, decode = encode.train(), middle.train(), decode.train()
	for step, pack in enumerate(trainloader):
		img, y, hd, imgp = pack
		if opt.gpu:
			img, y, hd = img.cuda(), y.cuda(), hd.cuda()

		out = encode(img)
		out = middle(out)
		out_chord, out_finger = decode(out)

		loss_chord = loss_func(out_chord, y)
		loss_finger = torch.sum((out_finger-hd)**2)
		loss = loss_chord + loss_finger
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		writer.add_scalar('train/chord_loss', loss_chord.item(), epoch*len(trainloader)+step)
		writer.add_scalar('train/finger_loss', loss_finger.item(), epoch*len(trainloader)+step)

	if epoch % opt.val_interval == 0:
		encode, middle, decode = encode.eval(), middle.eval(), decode.eval()
		with torch.no_grad():
			res, reg, gt = [], [], []
			for step, pack in enumerate(valloader):
				img, y, hd, imgp = pack
				if opt.gpu:
					img = img.cuda()

				out = encode(img)
				out = middle(out)
				out_chord, out_finger = decode(out)
				out_chord, out_finger = out_chord.cpu().numpy(), out_finger.cpu()

				res.append(out_chord)
				reg.append(torch.sum((out_finger-hd)**2).item())
				gt.append(y.numpy())

			res = np.concatenate(res, 0)
			res = np.argmax(res, 1)
			gt = np.concatenate(gt, 0)
			acc = np.mean(res == gt)
			writer.add_scalar('val/acc', acc, epoch)
			print('epoch:{} val acc:{} train chord loss:{} train finger loss:{}'.format(epoch, acc.item(), loss_chord.item(), loss_finger.item()))
			if best_val_acc <= acc:
				best_val_acc = acc
				torch.save({'encode': encode.module.state_dict() if opt.gpus else encode.state_dict(),
							'middle': middle.module.state_dict() if opt.gpus else middle.state_dict(),
							'decode': decode.module.state_dict() if opt.gpus else decode.state_dict(),
							'optimizer': optimizer}, os.path.join('./save', opt.name, 'check', 'best.pth.tar'))

		writer.flush()

writer.close()