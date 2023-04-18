import os
import numpy as np
import json
import glob
from tqdm import tqdm


def save_rel_path(filename):
    basename = os.path.basename(filename)
    v, g = basename.split('_')[1][4:], basename.split('_')[2][:2]
    v, g = int(v), int(g)
    cat = {
        'walk':{'1':[1,36], '2':[1,21], '3':[1,13], '4':[1,13], '5':[1,13],
                '6':[1,13], '7':[1,13], '8':[1,13], '9':[1,13], '10':[4,16]},
        'multi':{'1':[77,89], '2':[65, 77], '3':[13,22], '4':[13,25], '5':[13,25],
                '6':[13,25], '7':[13,25], '8':[13,25], '9':[13,25], '10':[16,28]},
        'styrofoam':{'1':[65,68], '2':[37, 45], '3':[22,30], '4':[25,33], '5':[25,33],
                '6':[25,33], '7':[25,33], '8':[25,33], '9':[25,33], '10':[28,36]},
        'carton':{'1':[62,65], '2':[21, 29], '3':[30,38], '4':[33,41], '5':[33,41],
                '6':[33,41], '7':[33,41], '8':[33,41], '9':[33,41], '10':[36,44]},
        'yoga':{'1':[0,0], '2':[29, 37], '3':[39,47], '4':[41,49], '5':[41,49],
                '6':[41,49], '7':[41,49], '8':[41,49], '9':[41,49], '10':[44,52]},
        'dark':{'1':[0,0], '2':[0, 0], '3':[0,0], '4':[0,0], '5':[0,0],
                '6':[0,0], '7':[0,0], '8':[61,71], '9':[61,72], '10':[64,74]},
        'action':{'1':[37,62], '2':[45,65], '3':[47,62], '4':[49,61], '5':[49,61],
                    '6':[49,63], '7':[49,61], '8':[49,61], '9':[49,61], '10':[52,64]}
    }
    if g == 0:
        path = None
    elif v == 10 and g in range(1, 4):
        # 排除第10组无人机数据
        path = None
    elif v == 3 and g == 38:
        # 补充第三组武治随意行走
        path = os.path.join('WALK', 'ANNOTATIONS', '2DPOSE', '%02d_%02d' % (v, g))
    elif v == 2 and g in range(77, 89):
        # 补充第二组多人遮挡
        # path = os.path.join('OCCLUSION', 'ANNOTATIONS', '2DPOSE', '%02d_%02d' % (v, g))
        path = None
    elif v == 1 and g in [36]:
        # 排除空场地
        path = None
    elif v == 1 and (g in range(73, 77)):
        # 补充双人泡沫塑料遮挡小涵烩面，双人箱子遮挡小涵烩面
        # path = os.path.join('OCCLUSION', 'ANNOTATIONS', '2DPOSE', '%02d_%02d' % (v, g))
        path = None
    elif v == 1 and (g in range(68, 73) or g in range(95, 98)):
        # 排除白板遮挡
        path = None
    elif v == 1 and g in range(89, 95):
        # 补充小涵箱子遮挡，泡沫塑料遮挡武治
        # path = os.path.join('OCCLUSION', 'ANNOTATIONS', '2DPOSE', '%02d_%02d' % (v, g))
        path = None
    elif g in range(*cat['walk'][str(v)]):
        path = os.path.join('WALK', 'ANNOTATIONS', '2DPOSE', '%02d_%02d' % (v, g))
    elif g in range(*cat['multi'][str(v)]):
        path = os.path.join('MULTI', 'ANNOTATIONS', '2DPOSE', '%02d_%02d' % (v, g))
    elif g in range(*cat['action'][str(v)]):
        path = os.path.join('ACTION', 'ANNOTATIONS', '2DPOSE', '%02d_%02d' % (v, g))
    elif g in list(range(*cat['styrofoam'][str(v)])) + list(range(*cat['carton'][str(v)])) + list(range(*cat['yoga'][str(v)])):
        # path = os.path.join('OCCLUSION', 'ANNOTATIONS', '2DPOSE', '%02d_%02d' % (v, g))
        path = None
    elif g in range(*cat['dark'][str(v)]):
        # path = os.path.join('DARK', 'ANNOTATIONS', '2DPOSE', '%02d_%02d' % (v, g))
        path = None
    else:
        assert False
    return path

in_prefix = '/mnt/hdd/hiber/seg/out'
out_prefix = '/mnt/hdd/hiber'

files = glob.glob(os.path.join(in_prefix, '*', 'alphapose-results.json'))

for anno_file in tqdm(files):
    with open(anno_file) as f:
        annos = json.load(f)
    out = save_rel_path(anno_file.split('/')[-2])
    grp_keypoints = []
    for i in range(590):
        key = '%d.jpg' % i
        if not key in annos.keys():
            empty_kps = np.zeros((num_p, 14, 2))
            grp_keypoints.append(empty_kps)
            continue
        keypoints = annos[key]['people']
        keypoints = [x['pose_keypoints_2d'] for x in keypoints]
        keypoints = np.array(keypoints).reshape(-1, 18, 3)
        keypoints = keypoints[np.where(np.mean(keypoints[:, :, -1], axis=-1) > 0.7)]
        keypoints = keypoints[:2]
        keypoints = keypoints[:, :14, :2]
        grp_keypoints.append(keypoints)
        num_p = keypoints.shape[0]
    num_people = [x.shape[0] for x in grp_keypoints]
    p_count = np.argmax(np.bincount(num_people))
    for i in range(len(grp_keypoints)):
        if p_count > grp_keypoints[i].shape[0]:
            grp_keypoints[i] = np.concatenate([grp_keypoints[i], np.zeros((p_count - grp_keypoints[i].shape[0], 14, 2))], axis=0)
        elif p_count < grp_keypoints[i].shape[0]:
            grp_keypoints[i] = grp_keypoints[i][:p_count]
    grp = np.stack(grp_keypoints, axis=0)
    np.save(os.path.join(out_prefix, out + '.npy'), grp)