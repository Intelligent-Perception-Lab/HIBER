import json
from sys import prefix
from matplotlib import image
import numpy as np
import os

out_prefix = '/mnt/hdd/hiber'
in_dir = '/mnt/hdd/hiber/temp/corrected_Annotations'

def format_annos(annotation_files):

    hor_boxes = {}
    ver_boxes = {}
    keypoints = {}
    for annotation_file in annotation_files:
        with open(annotation_file) as f:
            annotations = json.load(f)['annotations']

        for anno in annotations:
            image_id = anno['image_id']
            bbox = anno['bbox']
            keypoint = anno['keypoints']

            v, g, f = int(image_id[:2]), int(image_id[2:6]), int(image_id[6:10])
            key = '%02d_%02d.npy' % (v, g)
            hor_boxes[key] = hor_boxes.get(key, [])
            ver_boxes[key] = ver_boxes.get(key, [])
            keypoints[key] = keypoints.get(key, [])

            hor_boxes[key].extend(bbox[:4])
            ver_boxes[key].extend(bbox[4:8])
            keypoints[key].extend(keypoint)
    return hor_boxes, ver_boxes, keypoints


anno_files = os.listdir(in_dir)
anno_files = [os.path.join(in_dir, x) for x in anno_files]
hor_boxes, ver_boxes, keypoints = format_annos(anno_files)

def save_rel_path(filename):
    v, g = int(filename[:2]), int(filename[3:5])
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
    if v == 1 and g in range(89, 95):
        path3d =  os.path.join('OCCLUSION', 'ANNOTATIONS', '3DPOSE')
        pathhbox = os.path.join('OCCLUSION', 'ANNOTATIONS', 'BOUNDINGBOX', 'HORIZONTAL')
        pathvbox = os.path.join('OCCLUSION', 'ANNOTATIONS', 'BOUNDINGBOX', 'VERTICAL')
    elif g in range(*cat['walk'][str(v)]):
        path3d =  os.path.join('WALK', 'ANNOTATIONS', '3DPOSE')
        pathhbox = os.path.join('WALK', 'ANNOTATIONS', 'BOUNDINGBOX', 'HORIZONTAL')
        pathvbox = os.path.join('WALK', 'ANNOTATIONS', 'BOUNDINGBOX', 'VERTICAL')
    elif g in range(*cat['multi'][str(v)]):
        path3d =  os.path.join('MULTI', 'ANNOTATIONS', '3DPOSE')
        pathhbox = os.path.join('MULTI', 'ANNOTATIONS', 'BOUNDINGBOX', 'HORIZONTAL')
        pathvbox = os.path.join('MULTI', 'ANNOTATIONS', 'BOUNDINGBOX', 'VERTICAL')
    elif g in range(*cat['action'][str(v)]):
        path3d =  os.path.join('ACION', 'ANNOTATIONS', '3DPOSE')
        pathhbox = os.path.join('ACION', 'ANNOTATIONS', 'BOUNDINGBOX', 'HORIZONTAL')
        pathvbox = os.path.join('ACION', 'ANNOTATIONS', 'BOUNDINGBOX', 'VERTICAL')
    elif g in list(range(*cat['styrofoam'][str(v)])) + list(range(*cat['carton'][str(v)])) + list(range(*cat['yoga'][str(v)])):
        path3d =  os.path.join('OCCLUSION', 'ANNOTATIONS', '3DPOSE')
        pathhbox = os.path.join('OCCLUSION', 'ANNOTATIONS', 'BOUNDINGBOX', 'HORIZONTAL')
        pathvbox = os.path.join('OCCLUSION', 'ANNOTATIONS', 'BOUNDINGBOX', 'VERTICAL')
    elif g in range(*cat['dark'][str(v)]):
        path3d =  os.path.join('DARK', 'ANNOTATIONS', '3DPOSE')
        pathhbox = os.path.join('DARK', 'ANNOTATIONS', 'BOUNDINGBOX', 'HORIZONTAL')
        pathvbox = os.path.join('DARK', 'ANNOTATIONS', 'BOUNDINGBOX', 'VERTICAL')
    else:
        assert False
    return path3d, pathhbox, pathvbox



# for key, value in hor_boxes.items():
#     path3d, pathhbox, pathvbox = save_rel_path(key)
#     boxes = np.array(value).reshape(590, -1, 4)
#     np.save(os.path.join(out_prefix, pathhbox, key), boxes)

# for key, value in ver_boxes.items():
#     path3d, pathhbox, pathvbox = save_rel_path(key)
#     boxes = np.array(value).reshape(590, -1, 4)
#     np.save(os.path.join(out_prefix, pathvbox, key), boxes)

for key, value in keypoints.items():
    path3d, pathhbox, pathvbox = save_rel_path(key)
    kps = np.array(value).reshape(590, -1, 14, 4)
    kps = kps[:, :, :, :3]
    # the following code change image coordinate to radar coordinate by substract （100, 160）
    # above results have experienced coordinates transform by -y-160*0.05
    # but that makes y negative, we actually want -(y-160*0.05) which could make y positive
    # therefore we undo and redo this procedure by subtract np.array([100 * 0.05, 160 * 0.05 - 16])
    kps[:, :, :, [0, 1]] = kps[:, :, :, [0, 1]] - np.array([100 * 0.05, 160 * 0.05 - 16])
    np.save(os.path.join(out_prefix, path3d, key), kps)