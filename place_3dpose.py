import glob
import os
import json
import numpy as np
import HIBERTools as hiber
from tqdm import tqdm

# kp_jsons = '/mnt/hdd/data/view*/keypoint/train_view*_*.json'
# kp_jsons = glob.glob(kp_jsons)

kp_jsons = []
prefix = '/mnt/hdd/hiber/rgbs/chunyang/home/RFPose/output/no_smooth_results'
for v in range(1, 11):
    g = 1
    while True:
        file_name = os.path.join(prefix, 'train_view%d_%02d.json' % (v, g))
        file_name_ext = file_name.replace('.json', '_multi_new.json')
        if os.path.exists(file_name_ext):
            kp_jsons.append(file_name_ext)
        elif os.path.exists(file_name):
            kp_jsons.append(file_name)
        else:
            if g > 100:
                break
        g = g + 1

out_root = '/mnt/hdd/hiber'

def get_class(v, g, statistics):
    key = None
    for k, value in statistics.items():
        if '%02d_%02d' % (v, g) in value:
            key = k
            return key
    return key

dataset = hiber.HIBERDataset('/mnt/hdd/hiber', subsets=['WALK'])

for kp_json in tqdm(kp_jsons):
    v, g = int(os.path.basename(kp_json).split('_')[1][4:]), int(os.path.basename(kp_json).split('_')[2][:2])
    
    out_prefix = get_class(v, g, dataset.statistics)
    if out_prefix is None:
        continue
    out_file = os.path.join(out_root, out_prefix, 'ANNOTATIONS', '3DPOSE', '%02d_%02d.npy' % (v, g))
    # if os.path.exists(out_file):
    #     continue

    with open(kp_json) as f:
        data = json.load(f)['annotations']
    kps = []
    for i in range(590):
        assert data[i]['frame'] == i
        kp = data[i]['keypoints']
        kp = np.array(kp).reshape(-1, 14, 3)
        kps.append(kp)
    kps = np.stack(kps, axis=0)
    np.save(out_file, kps)
