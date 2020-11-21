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
		img, y, imgp = pack
		if opt.gpu:
			img, y = img.cuda(), y.cuda()

		out = encode(img)
		out = middle(out)
		out = decode(out)

		loss = loss_func(out, y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		writer.add_scalar('train/loss', loss.item(),
						  epoch*len(trainloader)+step)
		print('epoch:{} step:{}/{} train loss:{:.4f}'.format(epoch,
														 step, len(trainloader), loss.item()))

	if epoch % opt.val_interval == 0:
		encode, middle, decode = encode.eval(), middle.eval(), decode.eval()
		with torch.no_grad():
			res, gt = [], []
			for step, pack in enumerate(valloader):
				img, y, imgp = pack
				if opt.gpu:
					img = img.cuda()

				out = encode(img)
				out = middle(out)
				out = decode(out).cpu().numpy()

				res.append(out)
				gt.append(y.numpy())

			res = np.concatenate(res, 0)
			res = np.argmax(res, 1)
			gt = np.concatenate(gt, 0)
			acc = np.mean(res == gt)
			writer.add_scalar('val/acc', acc, epoch)
			print('epoch:{} val acc:{}'.format(epoch, acc.item()))
			if best_val_acc <= acc:
				best_val_acc = acc
				torch.save({'encode': encode.module.state_dict() if opt.gpus else encode.state_dict(),
							'middle': middle.module.state_dict() if opt.gpus else middle.state_dict(),
							'decode': decode.module.state_dict() if opt.gpus else decode.state_dict(),
							'optimizer': optimizer}, os.path.join('./save', opt.name, 'check', 'best.pth.tar'))

			torch.save({'encode': encode.module.state_dict() if opt.gpus else encode.state_dict(),
						'middle': middle.module.state_dict() if opt.gpus else middle.state_dict(),
						'decode': decode.module.state_dict() if opt.gpus else decode.state_dict(),
						'optimizer': optimizer}, os.path.join('./save', opt.name, 'check', 'epoch_{}.pth.tar'.format(epoch)))

		writer.flush()

writer.close()
