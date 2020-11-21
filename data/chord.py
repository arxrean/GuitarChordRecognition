import os
import cv2
import pdb
import json
import glob
import time
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import urllib.request
from PIL import Image
from dateutil import parser
from datetime import datetime
from collections import Counter

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

import sys
sys.path.append('C:/Users/60205/Code/GuitarHandRecognition')
import utils


timezone_info = {
        "A": 1 * 3600,
        "ACDT": 10.5 * 3600,
        "ACST": 9.5 * 3600,
        "ACT": -5 * 3600,
        "ACWST": 8.75 * 3600,
        "ADT": 4 * 3600,
        "AEDT": 11 * 3600,
        "AEST": 10 * 3600,
        "AET": 10 * 3600,
        "AFT": 4.5 * 3600,
        "AKDT": -8 * 3600,
        "AKST": -9 * 3600,
        "ALMT": 6 * 3600,
        "AMST": -3 * 3600,
        "AMT": -4 * 3600,
        "ANAST": 12 * 3600,
        "ANAT": 12 * 3600,
        "AQTT": 5 * 3600,
        "ART": -3 * 3600,
        "AST": 3 * 3600,
        "AT": -4 * 3600,
        "AWDT": 9 * 3600,
        "AWST": 8 * 3600,
        "AZOST": 0 * 3600,
        "AZOT": -1 * 3600,
        "AZST": 5 * 3600,
        "AZT": 4 * 3600,
        "AoE": -12 * 3600,
        "B": 2 * 3600,
        "BNT": 8 * 3600,
        "BOT": -4 * 3600,
        "BRST": -2 * 3600,
        "BRT": -3 * 3600,
        "BST": 6 * 3600,
        "BTT": 6 * 3600,
        "C": 3 * 3600,
        "CAST": 8 * 3600,
        "CAT": 2 * 3600,
        "CCT": 6.5 * 3600,
        "CDT": -5 * 3600,
        "CEST": 2 * 3600,
        "CET": 1 * 3600,
        "CHADT": 13.75 * 3600,
        "CHAST": 12.75 * 3600,
        "CHOST": 9 * 3600,
        "CHOT": 8 * 3600,
        "CHUT": 10 * 3600,
        "CIDST": -4 * 3600,
        "CIST": -5 * 3600,
        "CKT": -10 * 3600,
        "CLST": -3 * 3600,
        "CLT": -4 * 3600,
        "COT": -5 * 3600,
        "CST": -6 * 3600,
        "CT": -6 * 3600,
        "CVT": -1 * 3600,
        "CXT": 7 * 3600,
        "ChST": 10 * 3600,
        "D": 4 * 3600,
        "DAVT": 7 * 3600,
        "DDUT": 10 * 3600,
        "E": 5 * 3600,
        "EASST": -5 * 3600,
        "EAST": -6 * 3600,
        "EAT": 3 * 3600,
        "ECT": -5 * 3600,
        "EDT": -4 * 3600,
        "EEST": 3 * 3600,
        "EET": 2 * 3600,
        "EGST": 0 * 3600,
        "EGT": -1 * 3600,
        "EST": -5 * 3600,
        "ET": -5 * 3600,
        "F": 6 * 3600,
        "FET": 3 * 3600,
        "FJST": 13 * 3600,
        "FJT": 12 * 3600,
        "FKST": -3 * 3600,
        "FKT": -4 * 3600,
        "FNT": -2 * 3600,
        "G": 7 * 3600,
        "GALT": -6 * 3600,
        "GAMT": -9 * 3600,
        "GET": 4 * 3600,
        "GFT": -3 * 3600,
        "GILT": 12 * 3600,
        "GMT": 0 * 3600,
        "GST": 4 * 3600,
        "GYT": -4 * 3600,
        "H": 8 * 3600,
        "HDT": -9 * 3600,
        "HKT": 8 * 3600,
        "HOVST": 8 * 3600,
        "HOVT": 7 * 3600,
        "HST": -10 * 3600,
        "I": 9 * 3600,
        "ICT": 7 * 3600,
        "IDT": 3 * 3600,
        "IOT": 6 * 3600,
        "IRDT": 4.5 * 3600,
        "IRKST": 9 * 3600,
        "IRKT": 8 * 3600,
        "IRST": 3.5 * 3600,
        "IST": 5.5 * 3600,
        "JST": 9 * 3600,
        "K": 10 * 3600,
        "KGT": 6 * 3600,
        "KOST": 11 * 3600,
        "KRAST": 8 * 3600,
        "KRAT": 7 * 3600,
        "KST": 9 * 3600,
        "KUYT": 4 * 3600,
        "L": 11 * 3600,
        "LHDT": 11 * 3600,
        "LHST": 10.5 * 3600,
        "LINT": 14 * 3600,
        "M": 12 * 3600,
        "MAGST": 12 * 3600,
        "MAGT": 11 * 3600,
        "MART": 9.5 * 3600,
        "MAWT": 5 * 3600,
        "MDT": -6 * 3600,
        "MHT": 12 * 3600,
        "MMT": 6.5 * 3600,
        "MSD": 4 * 3600,
        "MSK": 3 * 3600,
        "MST": -7 * 3600,
        "MT": -7 * 3600,
        "MUT": 4 * 3600,
        "MVT": 5 * 3600,
        "MYT": 8 * 3600,
        "N": -1 * 3600,
        "NCT": 11 * 3600,
        "NDT": 2.5 * 3600,
        "NFT": 11 * 3600,
        "NOVST": 7 * 3600,
        "NOVT": 7 * 3600,
        "NPT": 5.5 * 3600,
        "NRT": 12 * 3600,
        "NST": 3.5 * 3600,
        "NUT": -11 * 3600,
        "NZDT": 13 * 3600,
        "NZST": 12 * 3600,
        "O": -2 * 3600,
        "OMSST": 7 * 3600,
        "OMST": 6 * 3600,
        "ORAT": 5 * 3600,
        "P": -3 * 3600,
        "PDT": -7 * 3600,
        "PET": -5 * 3600,
        "PETST": 12 * 3600,
        "PETT": 12 * 3600,
        "PGT": 10 * 3600,
        "PHOT": 13 * 3600,
        "PHT": 8 * 3600,
        "PKT": 5 * 3600,
        "PMDT": -2 * 3600,
        "PMST": -3 * 3600,
        "PONT": 11 * 3600,
        "PST": -8 * 3600,
        "PT": -8 * 3600,
        "PWT": 9 * 3600,
        "PYST": -3 * 3600,
        "PYT": -4 * 3600,
        "Q": -4 * 3600,
        "QYZT": 6 * 3600,
        "R": -5 * 3600,
        "RET": 4 * 3600,
        "ROTT": -3 * 3600,
        "S": -6 * 3600,
        "SAKT": 11 * 3600,
        "SAMT": 4 * 3600,
        "SAST": 2 * 3600,
        "SBT": 11 * 3600,
        "SCT": 4 * 3600,
        "SGT": 8 * 3600,
        "SRET": 11 * 3600,
        "SRT": -3 * 3600,
        "SST": -11 * 3600,
        "SYOT": 3 * 3600,
        "T": -7 * 3600,
        "TAHT": -10 * 3600,
        "TFT": 5 * 3600,
        "TJT": 5 * 3600,
        "TKT": 13 * 3600,
        "TLT": 9 * 3600,
        "TMT": 5 * 3600,
        "TOST": 14 * 3600,
        "TOT": 13 * 3600,
        "TRT": 3 * 3600,
        "TVT": 12 * 3600,
        "U": -8 * 3600,
        "ULAST": 9 * 3600,
        "ULAT": 8 * 3600,
        "UTC": 0 * 3600,
        "UYST": -2 * 3600,
        "UYT": -3 * 3600,
        "UZT": 5 * 3600,
        "V": -9 * 3600,
        "VET": -4 * 3600,
        "VLAST": 11 * 3600,
        "VLAT": 10 * 3600,
        "VOST": 6 * 3600,
        "VUT": 11 * 3600,
        "W": -10 * 3600,
        "WAKT": 12 * 3600,
        "WARST": -3 * 3600,
        "WAST": 2 * 3600,
        "WAT": 1 * 3600,
        "WEST": 1 * 3600,
        "WET": 0 * 3600,
        "WFT": 12 * 3600,
        "WGST": -2 * 3600,
        "WGT": -3 * 3600,
        "WIB": 7 * 3600,
        "WIT": 9 * 3600,
        "WITA": 8 * 3600,
        "WST": 14 * 3600,
        "WT": 0 * 3600,
        "X": -11 * 3600,
        "Y": -12 * 3600,
        "YAKST": 10 * 3600,
        "YAKT": 9 * 3600,
        "YAPT": 10 * 3600,
        "YEKST": 6 * 3600,
        "YEKT": 5 * 3600,
        "Z": 0 * 3600,
}


def uniform_img_name():
    imgps = glob.glob(os.path.join('./dataset/chords/*/*'))
    for i, imgp in enumerate(imgps):
        name = imgp.split('/')[-1].split('.')[0]
        shutil.move(imgp, imgp.replace(name, str(i)))

def remove_all_npy_visual():
    removes = glob.glob(os.path.join('./dataset/chords/*/*_visual.*')) + glob.glob(os.path.join('./dataset/chords/*/*.npy'))
    for x in removes:
        os.remove(x)

def remove_joints_not_4():
    csv = pd.read_csv('./dataset/chords/guitar_chord_annotation2.csv')
    keep_idx = []
    for name, group in csv.groupby('HITId'):
        if len(group) != 1:
            times = list(group['SubmitTime'])
            times = [parser.parse(x, tzinfos=timezone_info) for x in times]
            keep_idx.append(list(group.index)[np.argmax(times)])
        else:
            keep_idx.append(list(group.index)[0])
    csv = csv[csv.index.isin(keep_idx)]
    csv = csv.replace(np.nan, '', regex=True)
    for idx in csv.index:
        path = csv.at[idx, 'Input.path']
        input_img_height, input_img_width = csv.at[idx, 'Answer.annotatedResult.inputImageProperties.height'], csv.at[idx, 'Answer.annotatedResult.inputImageProperties.width']
        points = eval(csv.at[idx, 'Answer.annotatedResult.keypoints'])
        if len(points) != 4:
            csv.at[idx, 'Reject'] = 'the number of joints must be 4'
        img = cv2.imread(path)
        height, width = img.shape[:2]
        if abs(height-input_img_height) > 2 or abs(width-input_img_width) > 2:
            csv.at[idx, 'Reject'] = 'As illustrated in the instruction, the image shouldn\'t be zoomed'

    csv.reset_index(drop=True, inplace=True)
    csv.to_csv('./dataset/chords/guitar_chord_annotation2.csv', index=False)


def extract_joints():
    csv = pd.read_csv('./dataset/chords/guitar_chord_annotation2.csv')
    csv = csv[csv['Reject'].isnull()]
    for idx in csv.index:
        path = csv.at[idx, 'Input.path']
        name = path.split('/')[-1]
        ext1, ext2 = name.split('.')
        input_img_height, input_img_width = csv.at[idx, 'Answer.annotatedResult.inputImageProperties.height'], csv.at[idx, 'Answer.annotatedResult.inputImageProperties.width']
        points = eval(csv.at[idx, 'Answer.annotatedResult.keypoints'])

        joints = [(x['x'], x['y']) for x in points]
        img = cv2.imread(path)
        img = utils.draw_hd(img, joints, size=None)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        np.save(path.replace(name, ext1+'_joint.npy'), joints)
        cv2.imwrite(path.replace(name, ext1+'_visual.'+ext2), img)


def filter_bad_result():
    csv = pd.read_csv('./dataset/chords/guitar_chord_annotation2.csv')
    right_visuals = glob.glob('./dataset/chords/*/*_visual.*')
    for vp in right_visuals:
        label = vp.split(os.sep)[-2]
        name = vp.split(os.sep)[-1]
        name = name.replace('_visual', '')
        ext1, ext2 = name.split('.')
        csv_path = './dataset/chords/{}/{}.{}'.format(label, ext1, ext2)
        csv.at[csv[csv['Input.path']==csv_path].index, 'Approve'] = 'x'

    reject_csv = csv[csv['Approve'].isnull()]
    for idx in reject_csv.index:
        path = csv.at[idx, 'Input.path']
        name = path.split('/')[-1]
        ext1, ext2 = name.split('.')
        npy_path = path.replace(name, ext1+'_joint.npy')
        if os.path.exists(npy_path):
            os.remove(npy_path)

    reject_csv = reject_csv[reject_csv['Reject'].isnull()]
    csv.at[reject_csv.index, 'Reject'] = 'Poor quality. The annotated points are not on the finger tips.'
    csv.to_csv('./dataset/chords/guitar_chord_annotation2.csv', index=False)


def check_visual_npy():
    right_visuals = glob.glob('./dataset/chords/*/*_visual.*')
    for vp in right_visuals:
        label = vp.split(os.sep)[-2]
        name = vp.split(os.sep)[-1]
        name = name.replace('_visual', '')
        ext1, ext2 = name.split('.')

        csv_path = './dataset/chords/{}/{}.{}'.format(label, ext1, ext2)
        npy_path = csv_path.replace(name, ext1+'_joint.npy')
        assert os.path.exists(npy_path)

class ChordLoader(Dataset):
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode
        self.chords = ['Am', 'F', 'C', 'Dm']

        data = sorted(glob.glob(os.path.join(opt.chord_root, '*', '*_visual.*')))
        label = [self.chords.index(x.split(os.sep)[-2]) for x in data]
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=40, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=40, random_state=42)

        prof_data = sorted(glob.glob(os.path.join(opt.chord_root, '*', 'prof_*.*')))
        prof_label = [self.chords.index(x.split(os.sep)[-2]) for x in prof_data]

        if mode == 'train':
            self.data = tuple(zip(X_train, y_train))
        elif mode == 'val':
            self.data = tuple(zip(X_val, y_val))
        else:
            self.data = tuple(zip(X_test, y_test))
            self.data += tuple(zip(prof_data, prof_label))

        self.trans = self.get_trans()

    # feat loader
    def __getitem__(self, idx):
        x, y = self.data[idx][0], self.data[idx][1]

        img = cv2.imread(x.replace('_visual', ''))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(img)
        img = self.trans(img)

        return img, y, x

    def __len__(self):
        return len(self.data)

    def get_trans(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.mode == 'train':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.5, saturation=(0.5, 1.5)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])


class ChordPointLoader(Dataset):
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode
        self.chords = ['Am', 'F', 'C', 'Dm']

        data = sorted(glob.glob(os.path.join(opt.chord_root, '*', '*_visual.*')))
        label = [self.chords.index(x.split(os.sep)[-2]) for x in data]
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=40, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=40, random_state=42)

        prof_data = sorted(glob.glob(os.path.join(opt.chord_root, '*', 'prof_*.*')))
        prof_data = [x for x in prof_data if 'Hm' not in x]
        prof_label = [self.chords.index(x.split(os.sep)[-2]) for x in prof_data]

        if mode == 'train':
            self.data = tuple(zip(X_train, y_train))
        elif mode == 'val':
            self.data = tuple(zip(X_val, y_val))
        else:
            self.data = tuple(zip(X_test, y_test))
            self.data += tuple(zip(prof_data, prof_label))

        self.trans = self.get_trans()

    # feat loader
    def __getitem__(self, idx):
        x, y = self.data[idx][0], self.data[idx][1]

        img = cv2.imread(x.replace('_visual', ''))
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hd = self.img2jointfile(x.replace('_visual', ''))
        if os.path.exists(hd):
            hd = np.load(hd)
        else:
            hd = np.zeros((4, 2))

        hd = hd / np.expand_dims(np.array([h, w]), 0)
        img = cv2.resize(img, (self.opt.img_resize, self.opt.img_resize))
        # hd = hd * self.opt.img_resize

        # for x, y in hd:
        #     img = cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 2)
        # cv2.imshow(self.data[idx][0], img)
        # cv2.waitKey(0)

        img = Image.fromarray(img)
        img = self.trans(img)

        return img, self.data[idx][1], hd, self.data[idx][0]

    def __len__(self):
        return len(self.data)

    def get_trans(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.mode == 'train':
            return transforms.Compose([
                transforms.ColorJitter(brightness=0.5, saturation=(0.5, 1.5)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])


    def img2jointfile(self, path):
        name = path.split(os.sep)[-1]
        ext1, ext2 = name.split('.')
        return path.replace(name, ext1+'_joint.npy')


class ChordMetricLoader(Dataset):
    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode
        self.chords = ['Am', 'F', 'C', 'Dm', 'Hm']

        self.prototype = []
        for x in self.chords:
            items = glob.glob(os.path.join(opt.chord_root, x, '*'))
            self.prototype.append((sorted([x for x in items if '_visual' not in x and '.npy' not in x])[0], self.chords.index(x)))


        data = sorted(glob.glob(os.path.join(opt.chord_root, '*', '*_visual.*')))
        label = [self.chords.index(x.split(os.sep)[-2]) for x in data]
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=40, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=40, random_state=42)

        prof_data = sorted(glob.glob(os.path.join(opt.chord_root, '*', 'prof_*.*')))
        prof_label = [self.chords.index(x.split(os.sep)[-2]) for x in prof_data]

        if mode == 'train':
            self.data = tuple(zip(X_train, y_train))
        elif mode == 'val':
            self.data = tuple(zip(X_val, y_val))
        else:
            self.data = tuple(zip(X_test, y_test))
            self.data += tuple(zip(prof_data, prof_label))
            self.data = [x for x in self.data if x not in self.prototype]

        self.trans = self.get_trans()
        self.prototype = self.stack_prototype()

    # feat loader
    def __getitem__(self, idx):
        x, y = self.data[idx][0], self.data[idx][1]

        img = cv2.imread(x.replace('_visual', ''))
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hd = self.img2jointfile(x.replace('_visual', ''))
        if os.path.exists(hd):
            hd = np.load(hd)
        else:
            hd = np.zeros((4, 2))

        hd = hd / np.expand_dims(np.array([h, w]), 0)
        img = cv2.resize(img, (self.opt.img_resize, self.opt.img_resize))

        # for x, y in hd:
        #     img = cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 2)
        # cv2.imshow(self.data[idx][0], img)
        # cv2.waitKey(0)

        img = Image.fromarray(img)
        img = self.trans(img)

        return img, self.data[idx][1], hd, self.data[idx][0]

    def __len__(self):
        return len(self.data)

    def get_trans(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.mode == 'train':
            return transforms.Compose([
                transforms.ColorJitter(brightness=0.5, saturation=(0.5, 1.5)),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])


    def img2jointfile(self, path):
        name = path.split(os.sep)[-1]
        ext1, ext2 = name.split('.')
        return path.replace(name, ext1+'_joint.npy')


    def stack_prototype(self):
        items, labels, path = [], [], []
        for idx in range(len(self.prototype)):
            x, y = self.prototype[idx][0], self.prototype[idx][1]
            img = cv2.imread(x.replace('_visual', ''))
            h, w, c = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.opt.img_resize, self.opt.img_resize))
            img = Image.fromarray(img)
            img = self.trans(img)
            items.append(img)
            labels.append(y)
            path.append(x)

        items = torch.stack(items)

        return {'prototype': items, 'path': path, 'label': labels}



if __name__ == '__main__':
    # uniform_img_name()
    # remove_all_npy_visual()
    # remove_joints_not_4()
    # remove_joints_zoom_image()
    # extract_joints()
    # filter_bad_result()
    check_visual_npy()
