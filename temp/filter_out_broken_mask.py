import numpy as np
import matplotlib.pyplot as plt
import glob
import HIBERTools as hiber
from tqdm import tqdm


files = glob.glob('/mnt/hdd/hiber/*/ANNOTATIONS/SILHOUETTE/*/*.npy')
files = sorted(files)
dataset = hiber.HIBERDataset('/mnt/hdd/hiber', subsets=['MULTI'])

def max_num(v, g, dataset):
    statistics = dataset.statistics
    details = dataset.details
    seq = '%02d_%02d' % (v, g)
    if seq in statistics['MULTI'] + details['occlusion_multi'] + details['dark_multi']:
        return 2
    else:
        return 1

for mask in tqdm(files):
    v, g = mask.split('/')[-2].split('_')
    v, g = int(v), int(g)
    num = max_num(v, g, dataset)
    if num == 2:
        continue
    arr = np.load(mask)
    n_mask = arr.shape[0]
    if n_mask <= num:
        continue
    else:
        arr = arr[:num]
        np.save(mask, arr)
        # pass
        # for i, m in enumerate(arr):
        #     plt.imshow(m)
        #     plt.savefig('mask_%d.jpg' % i)