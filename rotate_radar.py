import os
import glob
import numpy as np
import HIBERTools as hiber
from tqdm import tqdm

dataset = hiber.HIBERDataset('')
dataset.datas


prefix = '/mnt/hdd/data'
radars = glob.glob(os.path.join(prefix, 'view*', '*', '*', 'r*_*_*.npy'))
out_root = '/mnt/hdd/hiber'


HOR = 'HORIZONTAL_HEATMAPS'
VER = 'VERTICAL_HEATMAPS'

for radar in tqdm(radars):
    orien = HOR if radar.find('hor') != -1 else VER
    v = int(radar.split('/')[4][4:])
    basename = os.path.basename(radar).split('_')
    g, f = int(basename[1]), int(basename[2][:4])
    cate = None
    for key, value in dataset.statistics.items():
        if '%02d_%02d' % (v, g) in value:
            cate = key
            break
    if cate is None:
        continue
    offset = 6
    if f < offset:
        continue
    out = os.path.join(out_root, cate, orien, '%02d_%02d' % (v, g), '%04d.npy' % (f - offset))
    rf = np.load(radar)
    rf = np.rot90(rf)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, rf)
    # pass