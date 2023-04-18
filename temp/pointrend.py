from statistics import mode
import warnings
warnings.filterwarnings("ignore")
from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import os
from tqdm import tqdm
import cv2
import time


def load_pointrend_model(config_file='point_rend_r50_caffe_fpn_mstrain_3x_coco.py', checkpoint_file='point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    return model

# test a single image and show the results
# img = '409.png'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)

# b, m = result[0][-1], result[1][-1]
# for i in range(1, len(result[0])):
#     result[0][i] = b
#     result[1][i] = m

# visualize the results in a new window
# model.show_result(img, result)
# or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)


def inference_vid(vid_name, model, out):
    video = mmcv.VideoReader(vid_name)
    for i, frame in enumerate(video):
        boxes, results = inference_detector(model, frame)
        if i >= 590:
            break
        persons = results[0]
        os.makedirs(out, exist_ok=True)
        if len(persons) == 0:
            np.save(os.path.join(out, '%04d.npy' % i), np.zeros((0, 1248, 1640), dtype=bool))
            continue
        if len(persons) > 2:
            persons = persons[:2]
        persons = np.stack(persons, axis=0)
        np.save(os.path.join(out, '%04d.npy' % i), persons)


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
    if v == 10 and g in range(1, 4):
        # 排除第10组无人机数据
        path = None
    elif v == 3 and g == 38:
        # 补充第三组武治随意行走
        path = os.path.join('WALK', 'ANNOTATIONS', 'SILHOUETTE', '%02d_%02d' % (v, g))
    elif v == 2 and g in range(77, 89):
        # 补充第二组多人遮挡
        path = os.path.join('OCCLUSION', 'ANNOTATIONS', 'SILHOUETTE', '%02d_%02d' % (v, g))
    elif v == 1 and g in [36]:
        # 排除空场地
        path = None
    elif v == 1 and (g in range(73, 77)):
        # 补充双人泡沫塑料遮挡小涵烩面，双人箱子遮挡小涵烩面
        path = os.path.join('OCCLUSION', 'ANNOTATIONS', 'SILHOUETTE', '%02d_%02d' % (v, g))
    elif v == 1 and (g in range(68, 73) or g in range(95, 98)):
        # 排除白板遮挡
        path = None
    elif v == 1 and g in range(89, 95):
        # 补充小涵箱子遮挡，泡沫塑料遮挡武治
        path = os.path.join('OCCLUSION', 'ANNOTATIONS', 'SILHOUETTE', '%02d_%02d' % (v, g))
    elif g in range(*cat['walk'][str(v)]):
        path = os.path.join('WALK', 'ANNOTATIONS', 'SILHOUETTE', '%02d_%02d' % (v, g))
    elif g in range(*cat['multi'][str(v)]):
        path = os.path.join('MULTI', 'ANNOTATIONS', 'SILHOUETTE', '%02d_%02d' % (v, g))
    elif g in range(*cat['action'][str(v)]):
        path = os.path.join('ACTION', 'ANNOTATIONS', 'SILHOUETTE', '%02d_%02d' % (v, g))
    elif g in list(range(*cat['styrofoam'][str(v)])) + list(range(*cat['carton'][str(v)])) + list(range(*cat['yoga'][str(v)])):
        path = os.path.join('OCCLUSION', 'ANNOTATIONS', 'SILHOUETTE', '%02d_%02d' % (v, g))
    elif g in range(*cat['dark'][str(v)]):
        path = os.path.join('DARK', 'ANNOTATIONS', 'SILHOUETTE', '%02d_%02d' % (v, g))
    else:
        assert False
    return path


prefix = '/mnt/hdd/hiber/seg/view%d/'
# subdirs = ['20210630', '20210706', '20210710', '20210712', '20210715', '20210718', '20210719', '20210720', '20210721', '20210723']

out_prefix = '/mnt/hdd/hiber'

model = load_pointrend_model()

for i in range(1, 11):
    path = prefix % i
    vids = [os.path.join(path, x) for x in os.listdir(path) if not x.endswith('_00.mp4') and x.endswith('mp4')]
    vids = [x for x in vids if x.find('test') == -1]
    # vids = ['/mnt/hdd/hiber/seg/view1/train_view1_80.mp4']
    for vid in tqdm(vids):
        tgt_path = save_rel_path(vid)
        if tgt_path is None:
            continue
        inference_vid(vid, model, os.path.join(out_prefix, tgt_path))
        
