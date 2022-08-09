import glob
import os
import json
from tkinter import N
import numpy as np
import HIBERTools as hiber
from tqdm import tqdm

boxes = glob.glob('/mnt/hdd/data/view*/min_bbox15/*.npy')
dataset = hiber.HIBERDataset('/mnt/hdd/hiber')

def get_class(v, g, statistics):
    key = None
    for k, value in statistics.items():
        if '%02d_%02d' % (v, g) in value:
            key = k
            return key
    return key

prefix = '/mnt/hdd/hiber'


for box in tqdm(boxes):
    v = int(box.split('/')[4][4:])
    g = int(os.path.basename(box).split('/')[-1].split('_')[1][:4])
    out_prefix = get_class(v, g, dataset.statistics)
    if out_prefix is None:
        continue
    dst = os.path.join(prefix, out_prefix, 'ANNOTATIONS', 
                        'BOUNDINGBOX', 'HORIZONTAL', '%02d_%02d.npy' % (v, g))
    # if os.path.exists(dst):
    #     continue
    # else:
    bb = np.load(box).reshape(590, -1, 4)
    np.save(dst, bb)
