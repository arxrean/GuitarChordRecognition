import os
import cv2
import pdb
import glob
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import utils


def video2frame(dir='./dataset'):
    videos = glob.glob(os.path.join(dir, 'video2', '*'))
    for i, v in enumerate(videos):
        print(v)
        fpath = os.path.join('./dataset/frames/{}'.format(i))
        if not os.path.exists(fpath):
            os.makedirs(fpath)

        frames = utils.get_imgs_from_video(v)
        for i, f in enumerate(frames):
            h, w, c = f.shape
            f = f[h//3:, w//3:]
            cv2.imwrite(os.path.join(fpath, '{:06d}.jpg'.format(i)), f)


def crop_hand(d='./dataset/frames'):
    frames = glob.glob(os.path.join(d, '*', '*'))
    for f in frames:
        bbox = np.load(f.replace('frames', 'bbox').replace('.jpg', '.npz'))
        bbox, scores = bbox['boxes'], bbox['scores']
        if len(bbox) > 0:
            for i in range(len(bbox)):
                _bbox = bbox[i]
                img = cv2.imread(f)
                height, width, channels = img.shape
                _bbox = [int(math.floor(_bbox[0]*height)), int(math.floor(
                    _bbox[1]*width)), int(math.ceil(_bbox[2]*height)), int(math.ceil(_bbox[3]*width))]
                _bbox = utils.extend_bbox(img, _bbox)
                crop = img[_bbox[0]:_bbox[2], _bbox[1]:_bbox[3]]

                output_d = '/'.join(f.replace('frames', 'crop').split('/')[:-1])
                if not os.path.exists(output_d):
                    os.makedirs(output_d)

                cv2.imwrite(os.path.join(output_d, f.split('/')[-1][:-4]+'_{}{}'.format(i, f.split('/')[-1][-4:])), crop)


def extractNonduplicateFrames(d='./dataset/non-dup-right'):
    all_frames = sorted(glob.glob(os.path.join(d, '*.jpg')))
    videos = {}
    for f in all_frames:
        v = f.split('/')[-1].split('_')[0]
        if v not in videos:
            videos[v] = []
        videos[v].append(f)

    videos = list(videos.values())

    ansh = []
    for v in videos:
        for i, f in enumerate(v):
            fimg = cv2.imread(f)
            hist = cv2.calcHist([fimg], [0], None, [256], [0, 256])
            if i == 0:
                os.system(
                    'cp {} ./dataset/non-dup-right-non-dup2/{}'.format(f, f.split('/')[-1]))
            else:
                compare = cv2.compareHist(
                    hist, ansh[-1], cv2.HISTCMP_BHATTACHARYYA)
                if compare > 0.03:
                    os.system(
                        'cp {} ./dataset/non-dup-right-non-dup2/{}'.format(f, f.split('/')[-1]))

            ansh.append(hist)


def crop_right(d='./dataset/non-dup'):
    frames = glob.glob(os.path.join(d, '*.jpg'))
    for x in frames:
        print(x)
        img = cv2.imread(x)
        img = img[:, 640:]
        cv2.imwrite('./dataset/non-dup-right/{}'.format(x.split('/')[-1]), img)


def parse_amt_file(f='./repo/03_batch_results.csv'):
    csv = pd.read_csv(f)
    ids, hits, fingers_list = [], [], [[], [], [], [], []]
    fingers = ['index', 'middle', 'ring', 'pinky', 'thumb']
    for idx in csv.index:
        url = csv.at[idx, 'Input.image_url']
        id = url.split('?')[0].split('/')[-1][:-4]
        hit = csv.at[idx, 'HITId']
        res = []
        for f in fingers:
            for s in ['6', '5', '4', '3', '2', '1']:
                col = 'Answer.{}-{}.on'.format(f, s)
                val = csv.at[idx, col]
                res.append(val)

        res = [res[i:i+6] for i in range(0, len(res), 6)]
        for i in range(len(fingers_list)):
            fingers_list[i].append(res[i])

        ids.append(id)
        hits.append(hit)

    rcsv = pd.DataFrame({'id': ids,
                         'hit': hits,
                         'index': fingers_list[0],
                         'middle': fingers_list[1],
                         'ring': fingers_list[2],
                         'pinky': fingers_list[3],
                         'thumb': fingers_list[4]})

    rcsv.to_csv('./repo/03.csv', index=False)


def train_val_split(d='./dataset/non-dup-right-non-dup2'):
    frames = glob.glob(os.path.join(d, '*.jpg'))
    piece = len(frames) // 10
    trainval, test = train_test_split(
        frames, test_size=piece*3, random_state=42)
    train, val = train_test_split(trainval, test_size=piece*2, random_state=42)
    np.savez('./repo/split.npz', train=train, val=val, test=test)

def convert_all_to_jpg(root='./dataset/chords'):
    images = glob.glob(os.path.join(root, '*', '*'))
    for imgp in images:
        name = imgp.split('/')[-1].split('.')[0]
        ext = '.'+imgp.split('/')[-1].split('.')[-1]
        img = cv2.imread(imgp)
        cv2.imwrite(imgp.replace(ext, '.jpg'), img)
        os.remove(imgp)


def chord2csv(folder='./dataset/chords'):
    imgs = glob.glob(os.path.join(folder, '*', '*'))
    items = []
    for img in imgs:
        label = img.split('/')[-2]
        name = img.split('/')[-1]
        items.append([name, img, label])

    csv = pd.DataFrame(items, columns=['name', 'path', 'label'])
    csv.to_csv('./repo/chord.csv', index=False)


def upload_img_s3():
    import boto3

    REGION = 'us-east-2'
    ACCESS_KEY_ID = 'XXX'
    SECRET_ACCESS_KEY = 'XXX'

    BUCKET_NAME = 'guitarchord2'

    csv = pd.read_csv('./repo/chord.csv')
    s3_client = boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY
    )
    for idx in csv.index:
        name = csv.at[idx, 'name']
        path = csv.at[idx, 'path']
        assert os.path.exists(path)
        response = s3_client.upload_file(path, BUCKET_NAME, name)


def gen_s3_url(outpath='./repo/s3_urls.csv', DATA_ROOT='./dataset/new', addon=False):
    import boto3

    REGION = 'us-east-2'
    ACCESS_KEY_ID = 'XXX'
    SECRET_ACCESS_KEY = 'XXX'

    BUCKET_NAME = 'guitarchord2'

    csv = pd.read_csv('./repo/chord.csv')
    csv['s3'] = ''

    s3c = boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY
    )

    session = boto3.Session(
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY,
    )
    s3_resource = session.resource('s3')

    my_bucket = s3_resource.Bucket(BUCKET_NAME)

    items = []
    for file in my_bucket.objects.all():
        params = {'Bucket': BUCKET_NAME, 'Key': file.key}
        url = s3c.generate_presigned_url('get_object', params).split('?')[0]
        name = url.split('/')[-1]
        csv.at[csv[csv['name']==name].index, 's3'] = url
        assert len(csv[csv['name']==name]) > 0

    visuals = glob.glob('./dataset/chords/*/*_visual.*')
    names = [x.split(os.sep)[-1].replace('_visual', '') for x in visuals]
    csv = csv[~csv['name'].isin(names)]
    csv.to_csv('./repo/chord2.csv', index=False)


if __name__ == '__main__':
    # video2frame()
    # crop_hand()
    # extractNonduplicateFrames()
    # crop_right()
    # parse_amt_file()
    # train_val_split()
    # parse_chord_csv()
    # convert_all_to_jpg()
    # chord2csv()
    # upload_img_s3()
    gen_s3_url()
