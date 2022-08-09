from torch import rand
import HIBERTools.HIBERTools as hiber
import glob
import os
import random
import lmdb
import numpy as np


dataset = hiber.HIBERDataset('/mnt/hdd/hiber', subsets=['WALK'])
# seq = random.randint(0, len(dataset) - 1)
# seq = 6000
# for i in range(50):
#     seq = random.randint(0, len(dataset) - 1)
#     data_item = dataset[seq + i]
#     hiber.visualize(data_item, 'test.jpg')
# dataset.save_as_lmdb('hiber.lmdb')


# prefix = '/mnt/hdd/hiber/temp/corrected_Annotations/*.json'

# json_files = glob.glob(prefix)

# seqs = []
# for json_file in json_files:
#     seq = os.path.basename(json_file).split('.')[0].split('_')[-1]
#     v, g = int(seq[:2]), int(seq[2:])
#     seqs.append('%02d_%02d' % (v, g))

# res = set(seqs) - set(dataset.statistics['WALK'])
keys = dataset.get_lmdb_keys()
env = lmdb.open('hiber.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    key = keys[0]
    data_item = []
    for k in key:
        buf = txn.get(k.encode('ascii'))
        if k.startswith('m'):
            data = np.frombuffer(buf, dtype=bool)
        else:
            data = np.frombuffer(buf, dtype=np.float64)
        data_item.append(data)
    data_item[0] = data_item[0].reshape(160, 200, 2)
    data_item[1] = data_item[1].reshape(160, 200, 2)
    data_item[2] = data_item[2].reshape(-1, 14, 2)
    data_item[3] = data_item[3].reshape(-1, 14, 3)
    data_item[4] = data_item[4].reshape(-1, 4)
    data_item[5] = data_item[5].reshape(-1, 4)
    data_item[6] = data_item[6].reshape(-1, 1248, 1640)
    hiber.visualize(data_item, 'lmdb.jpg')