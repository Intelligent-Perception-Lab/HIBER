import glob
import json
import os
import cv2
import numpy as np
import HIBERTools as hiber



#====================================================================================
# create radar frame link file

# prefix = '/mnt/hdd/data'
# h_pattern = 'r15_%04d_%04d.npy'
# v_pattern = 'r139_%04d_%04d.npy'

# out_prefix = '/mnt/hdd/hiber'

# views = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# groups = {
#             'walk':{'1':[1,36], '2':[1,21], '3':[1,13], '4':[1,13], '5':[1,13],
#                     '6':[1,13], '7':[1,13], '8':[1,13], '9':[1,13], '10':[4,16]},
#             'multi':{'1':[77,89], '2':[65, 77], '3':[13,22], '4':[13,25], '5':[13,25],
#                     '6':[13,25], '7':[13,25], '8':[13,25], '9':[13,25], '10':[16,28]},
#             'styrofoam':{'1':[65,68], '2':[37, 45], '3':[22,30], '4':[25,33], '5':[25,33],
#                     '6':[25,33], '7':[25,33], '8':[25,33], '9':[25,33], '10':[28,36]},
#             'carton':{'1':[62,65], '2':[21, 29], '3':[30,38], '4':[33,41], '5':[33,41],
#                     '6':[33,41], '7':[33,41], '8':[33,41], '9':[33,41], '10':[36,44]},
#             'yoga':{'1':[0,0], '2':[29, 37], '3':[39,47], '4':[41,49], '5':[41,49],
#                     '6':[41,49], '7':[41,49], '8':[41,49], '9':[41,49], '10':[44,52]},
#             'dark':{'1':[0,0], '2':[0, 0], '3':[0,0], '4':[0,0], '5':[0,0],
#                     '6':[0,0], '7':[0,0], '8':[61,71], '9':[61,72], '10':[64,74]},
#             'action':{'1':[37,62], '2':[45,65], '3':[47,62], '4':[49,61], '5':[49,61],
#                         '6':[49,63], '7':[49,61], '8':[49,61], '9':[49,61], '10':[52,64]}
#         }

# key = 'yoga'
# KEY = 'OCCLUSION'
# for v in views:
#     for g in list(range(*groups[key][str(v)])):
#         for f in range(590):
#             offset = 6
#             hpath = os.path.join(prefix, 'view%d' % v, 'hor', '%04d' % g, h_pattern % (g, f + offset))
#             vpath = os.path.join(prefix, 'view%d' % v, 'ver', '%04d' % g, v_pattern % (g, f + offset))

#             hout0 = os.path.join(out_prefix, KEY, 'HORIZONTAL_HEATMAPS', '%02d_%02d' % (v, g), '%04d.npy' % f)
#             vout0 = os.path.join(out_prefix, KEY, 'VERTICAL_HEATMAPS', '%02d_%02d' % (v, g), '%04d.npy' % f)

#             hcmd = 'ln -s %s %s' % (hpath, hout0)
#             print(hcmd)
#             os.system(hcmd)

#             vcmd = 'ln -s %s %s' % (vpath, vout0)
#             print(vcmd)
#             os.system(vcmd)

            # assert False
#====================================================================================


#====================================================================================
# save remaining keypoints without human correction

def get_class(v, g, statistics):
    key = None
    for k, value in statistics.items():
        if '%02d_%02d' % (v, g) in value:
            key = k
            return key
    return key

kp_files = glob.glob('/mnt/hdd/data/view*/keypoint/train_view*_*.json')
out_root = '/mnt/hdd/hiber'

dataset = hiber.HIBERDataset('/mnt/hdd/hiber', subsets=['WALK'])

for kp_file in kp_files:
    v, g = int(os.path.basename(kp_file).split('_')[1][4:]), int(os.path.basename(kp_file).split('_')[2][:2])
    with open(kp_file) as f:
        data = json.load(f)['annotations']
    for i in range(590):
        assert data[i]['frame'] == i
        kp = data[i]['keypoints']
        kp = np.array(kp).reshape(-1, 3)

        out_prefix = get_class(v, g, dataset.statistics)
