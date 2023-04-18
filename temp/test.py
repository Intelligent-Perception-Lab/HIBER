from ast import walk
from cgi import test
import sys
sys.path.append('.')
import HIBERTools.HIBERTools as hiber
import numpy as np
import os
from tqdm import tqdm
import glob

dataset = hiber.HIBERDataset('/mnt/hdd/hiber')

in_prefix = '/mnt/hdd/hiber'
train_prefix = os.path.join(in_prefix, 'HIBER_TRAIN')
val_prefix = os.path.join(in_prefix, 'HIBER_VAL')
test_prefix = os.path.join(in_prefix, 'HIBER_TEST')

def move_or_split_dataset(data, in_prefix, out_prefix):

    for k, v in tqdm(data.items()):
        for g in v:
            src = os.path.join(in_prefix, k, 'HORIZONTAL_HEATMAPS', g, '*.npy')
            dst = os.path.join(out_prefix, k, 'HORIZONTAL_HEATMAPS', g)

            os.makedirs(dst, exist_ok=True)
            if len(glob.glob(src)) != 0:
                cmd = 'ln -s %s %s' % (src, dst)
                os.system(cmd)
            else:
                print('Source file or directory %s does not exists' % src)

            src = os.path.join(in_prefix, k, 'VERTICAL_HEATMAPS', g, '*.npy')
            dst = os.path.join(out_prefix, k, 'VERTICAL_HEATMAPS', g)

            os.makedirs(dst, exist_ok=True)
            if len(glob.glob(src)) != 0:
                cmd = 'ln -s %s %s' % (src, dst)
                os.system(cmd)
            else:
                print('Source file or directory %s does not exists' % src)

    for k, v in tqdm(data.items()):
        for g in v:
            src = os.path.join(in_prefix, k, 'ANNOTATIONS', '2DPOSE', g + '.npy')
            dst = os.path.join(out_prefix, k, 'ANNOTATIONS', '2DPOSE', g + '.npy')

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if len(glob.glob(src)) != 0:
                cmd = 'ln -s %s %s' % (src, dst)
                os.system(cmd)
            else:
                print('Source file or directory %s does not exists' % src)

            src = os.path.join(in_prefix, k, 'ANNOTATIONS', '3DPOSE', g + '.npy')
            dst = os.path.join(out_prefix, k, 'ANNOTATIONS', '3DPOSE', g + '.npy')

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if len(glob.glob(src)) != 0:
                cmd = 'ln -s %s %s' % (src, dst)
                os.system(cmd)
            else:
                print('Source file or directory %s does not exists' % src)

            src = os.path.join(in_prefix, k, 'ANNOTATIONS', 'BOUNDINGBOX', 'HORIZONTAL', g + '.npy')
            dst = os.path.join(out_prefix, k, 'ANNOTATIONS', 'BOUNDINGBOX', 'HORIZONTAL', g + '.npy')

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if len(glob.glob(src)) != 0:
                cmd = 'ln -s %s %s' % (src, dst)
                os.system(cmd)
            else:
                print('Source file or directory %s does not exists' % src)

            src = os.path.join(in_prefix, k, 'ANNOTATIONS', 'BOUNDINGBOX', 'VERTICAL', g + '.npy')
            dst = os.path.join(out_prefix, k, 'ANNOTATIONS', 'BOUNDINGBOX', 'VERTICAL', g + '.npy')

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if len(glob.glob(src)) != 0:
                cmd = 'ln -s %s %s' % (src, dst)
                os.system(cmd)
            else:
                print('Source file or directory %s does not exists' % src)

            src = os.path.join(in_prefix, k, 'ANNOTATIONS', 'SILHOUETTE', g, '*.npy')
            dst = os.path.join(out_prefix, k, 'ANNOTATIONS', 'SILHOUETTE', g)

            os.makedirs(dst, exist_ok=True)
            if len(glob.glob(src)) != 0:
                cmd = 'ln -s %s %s' % (src, dst)
                os.system(cmd)
            else:
                print('Source file or directory %s does not exists' % src)
    
    cmd = 'rm -rf %s' % os.path.join(out_prefix, '*', '*_HEATMAPS', '*', '0592.npy')
    os.system(cmd)
    cmd = 'rm -rf %s' % os.path.join(out_prefix, '*', '*_HEATMAPS', '*', '0591.npy')
    os.system(cmd)
    cmd = 'rm -rf %s' % os.path.join(out_prefix, '*', '*_HEATMAPS', '*', '0590.npy')
    os.system(cmd)

move_or_split_dataset(dataset.__train_set__, in_prefix, train_prefix)
move_or_split_dataset(dataset.__val_set__, in_prefix, val_prefix)
move_or_split_dataset(dataset.__test_set__, in_prefix, test_prefix)