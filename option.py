import argparse


def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--name', type=str, default='chord')
	parser.add_argument('--seed', type=int, default=7)
	parser.add_argument('--gpu', action='store_true')
	parser.add_argument('--gpus', action='store_true')

	parser.add_argument('--chord_root', type=str, default='dataset/chords')

	# data
	parser.add_argument('--dataset', type=str, default='chord')
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--img_resize', type=int, default=224)

	# train
	parser.add_argument('--epoches', type=int, default=80)
	parser.add_argument('--base_lr', type=float, default=1e-5)
	parser.add_argument('--weight_decay', type=float, default=2e-5)
	parser.add_argument('--dropout', type=float, default=0.0)
	parser.add_argument('--val_interval', type=float, default=1)
	parser.add_argument('--encode', type=str, default='resnet50')
	parser.add_argument('--middle', type=str, default='avgp')
	parser.add_argument('--decode', type=str, default='fc')

	opt = parser.parse_args()

	return opt