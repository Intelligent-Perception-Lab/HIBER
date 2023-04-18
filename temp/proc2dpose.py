import json
import numpy as np
import cv2
import os

prefix = '/mnt/hdd/hiber/rgbs/chunyang/home/RFPose/Annotations'
out_root = '/mnt/hdd/hiber'

subdirs = ['20210630', '20210706', '20210710', '20210712', '20210715', '20210718', '20210719', '20210720', '20210721', '20210723']
v2dir = lambda x : subdirs[x-1]
camidx = lambda x: [2, 1, 2, 2, 4, 4, 5, 5, 6, 6][x-1]

# data_file = '/mnt/hdd/hiber/rgbs/chunyang/home/RFPose/Annotations/20210630/train_view1_80_multi_new.json'

def save_rel_path(v, g):
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
    if v == 10 and g in range(1, 4):
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
        paht = None
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

def draw(points, canvas, color=(255, 255, 255)):
    points = points[:, :2].astype(int)
    for point in points:
        canvas = cv2.circle(canvas, tuple(point), 3, color, -1)
    return canvas

# with open(data_file) as f:
#     data = json.load(f)

# canvas = np.zeros((1248, 1640), dtype=np.uint8)
# frame_data = data['0']
# cam2 = frame_data[2]['keypoints']
# # cam12 = frame_data[12]['keypoints']
# cam2 = np.array(cam2).reshape(-1, 3)
# # cam12 = np.array(cam12).reshape(-1, 3)

# cam2_kp = draw(cam2, canvas)
# cv2.imwrite('cam2.jpg', cam2_kp)

# cam12_kp = draw(cam12, canvas, color=(255, 0, 0))
# cv2.imwrite('cam12.jpg', cam12_kp)

for v in range(3, 4):
    path = os.path.join(prefix, v2dir(v))
    json_files = os.listdir(path)
    json_files = [x for x in json_files if x.startswith('train')]
    data_files = []
    for i in range(1, 100):
        n1, n2, n3 = 'train_view%d_%02d.json' % (v, i), 'train_view%d_%02d_new.json' % (v, i), 'train_view%d_%02d_multi_new.json' % (v, i)
        if n3 in json_files:
            data_files.append(n3)
        elif n2 in json_files:
            data_files.append(n2)
        elif n1 in json_files:
            data_files.append(n1)
        else:
            break
    data_files = [os.path.join(path, x) for x in data_files]
    for df in data_files:
        with open(df) as f:
            data = json.load(f)
        if len(os.path.basename(df)) <= 20:
            g = int(os.path.basename(df)[-7:-5])
        elif len(os.path.basename(df)) <=24:
            g = int(os.path.basename(df)[-11:-9])
        else:
            g = int(os.path.basename(df)[-17:-15])
        out_prefix = save_rel_path(v, g)
        if out_prefix is None:
            continue
        grp_keypoints = []

        for frame_id in range(0, 590):
            if str(frame_id) in data.keys():
                frame_data = data[str(frame_id)]
                for cam in frame_data:
                    if cam['camera'] == camidx(v):
                        keypoints = cam['keypoints']
                        break
                    else:
                        keypoints = None
                keypoints = np.array(keypoints).reshape(-1, 18, 3)
                keypoints = keypoints[np.where(np.mean(keypoints[:, :, -1], axis=-1) > 0.5)]
                keypoints = keypoints[:, :14, :2]
                grp_keypoints.append(keypoints)
                num_people = keypoints.shape[0]
                # canvas = np.zeros((1248, 1640), dtype=np.uint8)
                # cam2_kp = draw(keypoints[0], canvas)
                # cv2.imwrite('test.jpg', cam2_kp)
                # pass
            else:
                # lack of keypoints, save empty array
                empty_kps = np.zeros((num_people, 14, 2))
                grp_keypoints.append(empty_kps)
        num_people = [x.shape[0] for x in grp_keypoints]
        p_count = np.argmax(np.bincount(num_people))
        for i in range(len(grp_keypoints)):
            grp_keypoints[i] = np.concatenate([grp_keypoints[i], np.zeros((p_count - grp_keypoints[i].shape[0], 14, 2))], axis=0)
        grp = np.stack(grp_keypoints, axis=0)
        np.save(os.path.join(out_root, out_prefix + '.npy'), grp)