import os
import time
import shutil
import pprint
import torch
import cv2
import numpy as np

import torch.nn as nn
import torchvision.transforms as transforms


def get_imgs_from_video(video):
    frames = []
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames


def get_video_from_imgs(frames, path):
    height, width, layers = frames[0].shape
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def extend_bbox(img, bbox):
    height, width, channels = img.shape
    b_h, b_w = bbox[2]-bbox[0], bbox[3]-bbox[1]
    eb_h, eb_w = b_h//10, b_w//10
    bbox = [bbox[0]-eb_h, bbox[1]-eb_w, bbox[2]+eb_h, bbox[3]+eb_w]

    bbox[0] = max(bbox[0], 0)
    bbox[1] = max(bbox[1], 0)
    bbox[2] = min(bbox[2], height)
    bbox[3] = min(bbox[3], width)

    return bbox


def get_dataset(options):
    if options.dataset == 'chord':
        from data.chord import ChordLoader
        dataset_train = ChordLoader(options, mode='train')
        dataset_val = ChordLoader(options, mode='val')
        dataset_test = ChordLoader(options, mode='test')
    elif options.dataset == 'chord_point':
        from data.chord import ChordPointLoader
        dataset_train = ChordPointLoader(options, mode='train')
        dataset_val = ChordPointLoader(options, mode='val')
        dataset_test = ChordPointLoader(options, mode='test')
    elif options.dataset == 'chord_metric':
        from data.chord import ChordMetricLoader
        dataset_train = ChordMetricLoader(options, mode='train')
        dataset_val = ChordMetricLoader(options, mode='val')
        dataset_test = ChordMetricLoader(options, mode='test')
    else:
        raise

    return (dataset_train, dataset_val, dataset_test)


def init_log_dir(opt):
    if os.path.exists(os.path.join('./save', opt.name)):
        shutil.rmtree(os.path.join('./save', opt.name))

    os.mkdir(os.path.join('./save', opt.name))
    with open(os.path.join('./save', opt.name, 'options.txt'), "a") as f:
        for k, v in vars(opt).items():
            f.write('{} -> {}\n'.format(k, v))
            print('{} -> {}\n'.format(k, v))

    os.mkdir(os.path.join('./save', opt.name, 'check'))
    os.mkdir(os.path.join('./save', opt.name, 'imgs'))
    os.mkdir(os.path.join('./save', opt.name, 'tb'))


def get_model(options):
    if options.encode == 'resnet50':
        from model.encode import Resnet50
        encode = Resnet50(options)
    else:
        raise

    if options.middle == 'pass':
        from model.encode import Pass
        middle = Pass(options)
    elif options.middle == 'avgp':
        from model.middle import AvgPool
        middle = AvgPool(options)
    elif options.middle == 'point':
        from model.middle import Point
        middle = Point(options)
    else:
        raise

    if options.decode == 'pass':
        from model.decode import Pass
        decode = Pass(options)
    elif options.decode == 'point':
        from model.decode import PointFC
        decode = PointFC(options)
    elif options.decode == 'fc':
        from model.decode import FC
        decode = FC(options)
    else:
        raise

    return encode, middle, decode


def draw_hd(img, hds, size):
    if size:
        img = cv2.resize(img, size)
        
    for x, y in hds:
        image = cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 2)

    return image


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        
    return res