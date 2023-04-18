"""
@author: Chunyang Xie
@contact: ChunyangXie@std.uestc.edu.cn
Code of processing videos to get human 3D keypoints using AlphaPose
"""

import json
import math
import os
import time
import argparse
import scipy.io as sio
from tqdm import tqdm
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem
from pymvg.plot_utils import plot_system
from scipy.signal import savgol_filter
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D



def unfold_camera_param_from_mat(camera):
    W = camera['width'][0, 0].astype('float')
    H = camera['height'][0, 0].astype('float')
    R = camera['R'].astype('float')
    T = camera['T'].astype('float').reshape(3, 1)
    K = camera['K'].astype('float')
    D = camera['D'][0].astype('float')
    P = camera['P'].astype('float')
    return W, H, R, T, K, D, P


def unfold_camera_param_from_dict(camera):
    W = np.array(camera['W'])
    H = np.array(camera['H'])
    R = np.array(camera['R'])
    T = np.array(camera['T'])
    K = np.array(camera['K'])
    D = np.array(camera['D'])
    P = np.array(camera['P'])
    return W, H, R, T, K, D, P


def transfer_camera_system_to_json(matfile):
    """
    transfer calibration result from mat file to json file
    :param matfile: calibration result filepath
    :return: json file which contains camera parameters
    """
    cam_system = sio.loadmat(matfile)['system']
    pymvg_cameras = {}
    for i in range(cam_system.size):
        W, H, R, T, K, D, P = unfold_camera_param_from_mat(cam_system[0, i])
        cam_name = 'cam_' + str(i)
        pymvg_cameras[cam_name] = {}
        pymvg_cameras[cam_name]['W'] = W.tolist()
        pymvg_cameras[cam_name]['H'] = H.tolist()
        pymvg_cameras[cam_name]['R'] = R.tolist()
        pymvg_cameras[cam_name]['T'] = T.tolist()
        pymvg_cameras[cam_name]['K'] = K.tolist()
        pymvg_cameras[cam_name]['D'] = D.tolist()
        pymvg_cameras[cam_name]['P'] = P.tolist()
    json_name = matfile.split('.')[0] + '.json'
    with open(json_name, 'w') as f:
        json.dump(pymvg_cameras, f, indent=4, sort_keys=True)
    return pymvg_cameras


def build_multi_camera_system_from_dict(camera_names, camera_params):
    '''
    bulid multi-camera system form camera params dict
    :param camera_names: list contains cameras to bulid system
    :param camera_params: dict contains all cameras' params
    :return: camera system
    '''
    pymvg_cameras = []
    for cam in camera_names:
        W, H, R, T, K, D, P = unfold_camera_param_from_dict(camera_params[cam])
        # proj_matrix = np.zeros((3, 4))
        # proj_matrix[:3, :3] = K
        # T = -np.matmul(R, T)
        # M = K.dot(np.concatenate((R, T), axis=1))

        camera = CameraModel.load_camera_from_M(
            P, width=W, height=H, name=cam, distortion_coefficients=D)
        # print('center:{}'.format(camera.get_camcenter()))
        pymvg_cameras.append(camera)
    return MultiCameraSystem(pymvg_cameras)


def cluster_points(ann, nviews=12, save_path=None):
    """
    transfer annotations to numpy array
    :param ann: Dict like {'0':[{'camera':0,'keypoints':[]...},{'camera':1...},{'camera':2...}],'1':[{},{}]...,'2':[{},{}],...}
    :param nviews: number of cameras.
    :param save_path: path to save the result.
    :return: ndarray of shape nframes x nviews x njoints x 3
    """
    # cam_cycle = [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]  # dirty store cycle camera system
    frame = ann.keys()
    # print(frame)
    nframes = len(frame)
    # print(nframes)
    njoints = 14
    points = np.zeros((nframes, nviews, njoints, 3))
    # skeleton_width = np.zeros((nframes, nviews, 1))  # measure person skeleton width
    # skeleton_score = np.zeros((nframes, nviews, 1))
    for f in frame:
        kps = ann[f]
        for view in range(len(kps)):
            if kps[view]['keypoints']:
                cam, kp = kps[view]['camera'], kps[view]['keypoints'][0]

                kp = np.array(kp[:njoints * 3], dtype=float).reshape((-1, 3))
                # skeleton_width[int(f), int(cam)] = np.linalg.norm(kp[2, 0:2]-kp[5, 0:2])  # shoulder width
                if int(f) < nframes:
                    points[int(f), int(cam), :, :] = kp

        # for i in range(0, 12):
        #     cam_n = i
        #     cam_nm1 = (i-1 + 12) % 12
        #     cam_np1 = (i+1) % 12
        #     cam_np6 = (i+6) % 12
        #     cam_np5 = (i+5) % 12
        #     cam_np7 = (i+7) % 12
        #     width_n = skeleton_width[int(f), cam_n]
        #     width_nm1 = skeleton_width[int(f), cam_nm1]
        #     width_np1 = skeleton_width[int(f), cam_np1]
        #     width_np6 = skeleton_width[int(f), cam_np6]
        #     width_np5 = skeleton_width[int(f), cam_np5]
        #     width_np7 = skeleton_width[int(f), cam_np7]
        #
        #     skeleton_score[int(f), i] = (width_n + width_nm1 + width_np1 + width_np6 + width_np5 + width_np7)/6  # average 6 cameras
        #
        # max_ind = np.argmax(skeleton_score[int(f)])  # find the wildest view
        # c_n = max_ind
        # c_nm1 = (max_ind - 1 + 12) % 12
        # c_np1 = (max_ind + 1) % 12
        # c_np6 = (max_ind + 6) % 12
        # c_np5 = (max_ind + 5) % 12
        # c_np7 = (max_ind + 7) % 12
        # accept = [c_n, c_nm1, c_np1, c_np5, c_np6, c_np7]
        # for n in range(nviews):
        #     if n in accept:
        #         continue
        #     else:
        #         points[int(f), n, :, 2] = 0.0  # set the unaccept view score as 0
    if save_path:
        np.save(save_path, points)
    return points


def filter_points(pose3d, reproj_error, reproj_thresh=40, window=31):
    """
    Using filter to smooth the 3D skeleton
    :param pose3d: triangulated 3D coordinate of one keypoint, ndarray of shape (frame x k x 3)
    :param reproj_error: reprojection error matrix, ndarray of shape (frame x k x 1)
    :param reproj_thresh: if point's reprojection error is bigger than this thresh, it will be treated as bad point.
    :param window: filter window length which must be odd.
    :return: smoothed 3D keypoints , ndarray of shape (frame x k x 3)
    """
    # filter points that beyond the physical boundary of [-3m<x<3m, 0m<y<8m,-1.5m<z<1.5m]
    # beyond_border = np.where((abs(pose3d[:, :, 0]) < 3) | (abs(pose3d[:, :, 1] - 4) < 4) | (abs(pose3d[:, :, 2]) < 1.5))
    # for bad_f, bad_j in list(zip(beyond_border[0], beyond_border[1])):
    #     pose3d[bad_f, bad_j] = pose3d[bad_f-1, bad_j-1]  # Assign the data of the previous frame to the current frame

    new_pose3d = np.zeros_like(pose3d)
    nframes, njoints = pose3d.shape[0], pose3d.shape[1]
    ind = np.where(reproj_error[:, :, 0] > reproj_thresh)
    bad_frame, bad_joint = list(ind)[0], list(ind)[1]
    inteval = int((window - 1) / 2)
    for f, j in list(zip(bad_frame, bad_joint)):
        start_frame = max(f - inteval, 0)
        end_frame = min(f + inteval, nframes - 1)
        max_ind = np.where(reproj_error[start_frame:end_frame, j, 0] < reproj_thresh)
        max_frame = max_ind[0] + start_frame
        if max_frame.size == 0:
            continue
        elif np.min(max_frame) < f < np.max(max_frame):
            # interpolate
            # print('interpolate...')
            max_x = pose3d[max_frame, j, 0]
            max_y = pose3d[max_frame, j, 1]
            max_z = pose3d[max_frame, j, 2]
            inter_x = interpolate.interp1d(max_frame, max_x)
            inter_y = interpolate.interp1d(max_frame, max_y)
            inter_z = interpolate.interp1d(max_frame, max_z)
            new_x, new_y, new_z = inter_x(f), inter_y(f), inter_z(f)
            pose3d[f, j, :] = np.array([new_x, new_y, new_z])
        else:
            # fit
            continue
            max_x = pose3d[max_frame, j, 0]
            max_y = pose3d[max_frame, j, 1]
            max_z = pose3d[max_frame, j, 2]
            fit_x_p = np.polyfit(max_frame, max_x, 1)
            fit_y_p = np.polyfit(max_frame, max_y, 1)
            fit_z_p = np.polyfit(max_frame, max_z, 1)
            fit_x = np.poly1d(fit_x_p)
            fit_y = np.poly1d(fit_y_p)
            fit_z = np.poly1d(fit_z_p)
            new_x, new_y, new_z = fit_x(f), fit_y(f), fit_z(f)
            pose3d[f, j, :] = np.array([new_x, new_y, new_z])
    # smooth
    for n in range(njoints):
        pose_x = pose3d[:, n, 0]
        pose_y = pose3d[:, n, 1]
        pose_z = pose3d[:, n, 2]

        new_x = savgol_filter(pose_x, window, 2)
        new_y = savgol_filter(pose_y, window, 2)
        new_z = savgol_filter(pose_z, window, 2)
        new_pose3d[:, n, 0] = new_x
        new_pose3d[:, n, 1] = new_y
        new_pose3d[:, n, 2] = new_z

    return new_pose3d


def compute_reproj_error(X3d, X2d, cam_system):
    '''
    compute mean reprojection error of one keypoint of one frame
    :param X3d: triangulated 3D coordinate of one keypoint
    :param X2d: origin 2D coordinates of one kind of keypoint
    :param cam_system: camera system
    :return: mean reprojection error
    '''
    reproj_error = 0
    for name, xy in X2d:
        cam = cam_system._cameras[name]
        x2d_reproj = cam.project_3d_to_pixel(X3d)
        dx = x2d_reproj - np.array(xy)
        reproj_error += np.sqrt(np.sum(dx ** 2, axis=1))
    reproj_error /= len(X2d)

    return reproj_error


# code from https://stackoverflow.com/a/18994296
def closestDistanceBetweenLines(a0, a1, b0, b1, clampAll=False, clampA0=False, clampA1=False, clampB0=False,
                                clampB1=False):
    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
    '''

    # If clampAll=True, set all clamps to True
    if clampAll:
        clampA0 = True
        clampA1 = True
        clampB0 = True
        clampB1 = True

    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = np.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if np.absolute(d0) < np.absolute(d1):
                        return a0, b0, np.linalg.norm(a0 - b0)
                    return a0, b1, np.linalg.norm(a0 - b1)


            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1, b0, np.linalg.norm(a1 - b0)
                    return a1, b1, np.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return None, None, np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = np.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = np.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, np.linalg.norm(pA - pB)


def triagulate_single(poses2d, camera_params, scale=1.0, thresh=0.7, top_k=12):
    """
    Triangulate 3d points into world coordinates from multi-view 2d poses
    Args:
        camera_params: a list of camera parameters, each corresponding to
                       one prediction in poses2d
        poses2d: ndarray of shape framexnxkx3, len(cameras) == n
    Returns:
        poses3d: ndarray of shape frame x k x 3
    """

    nframes, njoints = poses2d.shape[0], poses2d.shape[2]

    pose3d = np.zeros((nframes, njoints, 3))
    reproj_error_mat = np.zeros((nframes, njoints, 1))
    for frame in range(nframes):
        for k in range(njoints):
            points_2d_set = []
            cameras = []
            points_2d_topk = poses2d[frame, :, k, -1].argsort()[:: -1][0:top_k]
            for n in points_2d_topk:
                points_2d = poses2d[frame, n, k, :]

                if points_2d[-1] < thresh:  # ignore some bad keypoints
                    continue
                camera_name = 'cam_{}'.format(n)
                # debug
                # if frame > 295:
                #     print(frame, k, camera_name)
                cameras.append(camera_name)
                points_2d_set.append((camera_name, points_2d[:2]))

            if len(points_2d_set) > 1:  # triagulation
                camera_system = build_multi_camera_system_from_dict(cameras, camera_params)
                # print(points_2d_set)
                p3d = camera_system.find3d(points_2d_set)
                error = compute_reproj_error(p3d.reshape([-1, 3]), points_2d_set, camera_system)
                reproj_error_mat[frame, k, :] = error
                # coordinate transformation
                p3d *= scale
                p3d_world = np.zeros(3)
                # p3d_world[0] = p3d[0]
                # p3d_world[1] = p3d[2]
                # p3d_world[2] = -p3d[1]
                p3d_world[0] = p3d[0]
                p3d_world[1] = p3d[1]
                p3d_world[2] = p3d[2]
                pose3d[frame, k, :] = p3d_world.T

            else:  # just copy the previous keypoints
                pose3d[frame, k, :] = pose3d[frame-1, k, :]

    return pose3d, reproj_error_mat


def plot3D(poses, colors, dataset='open', title=None):
    """Plot the 3D pose showing the joint connections.
    kp_names = ['nose', 'neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder',  # 5
                    'LElbow', 'LWrist', 'MidHip', 'RHip', 'RKnee',  # 10
                    'RAnkle', 'LHip', 'LKnee', 'LAnkle']
    poses : list of shape [ndarray(3,15), ndarray(3,15), ndarray(3,15)...]
    """
    if dataset is 'open':
        _CONNECTION = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
                       [10, 11], [8, 12], [12, 13], [13, 14]]
    elif dataset is 'alpha':
        _CONNECTION = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10],
                       [11, 12], [12, 13], [1, 8], [1, 11]]
    fig = plt.figure()

    ax = fig.gca(projection='3d')

    smallest = [min([i[idx].min() for i in poses]) for idx in range(3)]
    largest = [max([i[idx].max() for i in poses]) for idx in range(3)]
    ax.set_xlim3d(smallest[0], largest[0])
    ax.set_ylim3d(smallest[1], largest[1])
    ax.set_zlim3d(smallest[2], largest[2])
    ax.set_xlim3d(-4, 4)
    ax.set_ylim3d(0, 8)
    ax.set_zlim3d(-1.5, 1.5)
    if title:
        ax.set_title(label=title, loc='center')

    for i, pose in enumerate(poses):
        assert (pose.ndim == 2)
        assert (pose.shape[0] == 3)
        for c in _CONNECTION:  # plot lines
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.plot([pose[0, c[0]], pose[0, c[1]]],
                    [pose[1, c[0]], pose[1, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]], c=col)
        for j in range(pose.shape[1]):  # plot keypoints
            col = '#%02x%02x%02x' % (colors[i][0], colors[i][1], colors[i][2])
            ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                       c=col, marker='o', edgecolor=col)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

    return fig


def save_coco(pose3d, filename, imid_start, annid_start, nsample, interval=4, height=160, width=161):
    """
    save 3D pose results into COCO_like json file
    :param pose3d: ndarray of shape frame x k x 3
    :param filename: RF data filename like ""train_200525sce1sin1.json""
    :param nsample: number of sample frame. For example, nsample should be set as 166 if
                    RF device receive 166 packages at each experiment
    :param interval: if we need 9 frames to predict the middle 5th frame, interval should be set as 4.
    :param height: height of RF data matrix
    :param width: width of RF data matrix
    :return: coco-like annotation file
    """
    result = dict()
    result['images'] = []
    result['annotations'] = []
    result['category'] = {}

    result['category']['supercategory'] = 'person'
    result['category']['id'] = 1
    result['category']['name'] = 'person'
    result['category']['keypoints'] = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder',  # 5
                                       'LElbow', 'LWrist', 'RHip', 'RKnee', 'RAnkle',  # 10
                                       'LHip', 'LKnee', 'LAnkle']

    result['category']['skeleton'] = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10],
                                      [11, 12], [12, 13], [1, 8], [1, 11]]

    Prefix = filename.split('.')[0]  # "train_200525sce1sin1"

    nframe, njoints = pose3d.shape[0], pose3d.shape[1]
    imid, annid = imid_start, annid_start
    for i in range(nsample):
        im_name = "%04d.npy" % i
        im = dict()
        im['file_name'] = Prefix + '_' + im_name  # "train_0821wifiex71_0001.npy"
        im['height'] = height
        im['width'] = width
        im['id'] = imid  # to be modified
        result['images'].append(im)
        res = dict()
        frame = i  # don't sample frames
        res['keypoints'] = list(pose3d[frame].reshape(-1))
        res['num_keypoints'] = 14
        res['id'] = annid
        res['image_id'] = imid  # to be modified
        res['frame'] = frame
        result['annotations'].append(res)
        imid += 1
        annid += 1
    # origin
    # for i in range(nsample):
    #     if i < interval or i >= nsample - interval:
    #         continue
    #     im_name = "%04d.npy" % i
    #     im = dict()
    #     im['file_name'] = Prefix + '_' + im_name  # "train_0821wifiex71_0001.npy"
    #     im['height'] = height
    #     im['width'] = width
    #     im['id'] = imid  # to be modified
    #     result['images'].append(im)
    #     res = dict()
    #     frame = int(i * (nframe / nsample))  # how to sample frames
    #     res['keypoints'] = list(pose3d[frame].reshape(-1))
    #     res['num_keypoints'] = 14
    #     res['id'] = annid
    #     res['image_id'] = imid  # to be modified
    #     res['frame'] = frame
    #     result['annotations'].append(res)
    #     imid += 1
    #     annid += 1

    return result, imid, annid

def project2img(keypoints3d, cam_index=0, pad=25):
    #cam_file = '/data/TIPoseData/ti_data_process/projection/result-200828cal3.json'
    cam_file = 'view10_cam6_cal.json'
    with open(cam_file) as f:
        cam_data = json.load(f)
    cam_names = ['cam_%d' % x for x in range(13)]
    cameras = []
    cam_system = build_multi_camera_system_from_dict(cam_names, cam_data)        
    cam = cam_system._cameras[cam_names[cam_index]]
    #kp = '/data/TIPoseData/recrop200829/keypoint/0001/0200.npy'
    #kp = np.load(kp).reshape(-1, 3)
    kp = keypoints3d.reshape(-1, 3)
    #kp = change_view(kp, 6, cam_file)
    kp2d = cam.project_3d_to_pixel(kp)
    kp2d = kp2d / 2
    # return padded corners
    xmin, xmax = np.min(kp2d[:, 0]), np.max(kp2d[:, 0])
    ymin, ymax = np.min(kp2d[:, 1]), np.max(kp2d[:, 1])
    xmin, xmax = int(xmin) - pad, int(xmax) + pad
    ymin, ymax = int(ymin) - pad, int(ymax) + pad
    
    #return xmin, xmax, ymin, ymax, kp2d
    return kp2d, (xmin, ymin, xmax, ymax)

def draw_points2d(kp2d, shape=(624, 820, 3), canvas = None):
    if canvas is None:
        canvas = np.zeros((624, 820, 3), dtype=np.uint8)
    for k in kp2d:
        x, y = k
        center = (int(x), int(y))
        cv2.circle(canvas, center, 3, (255, 255, 255), 2)
    return canvas
    

def bbox3d(kp3d):
    kp3d = kp3d.reshape(-1, 3)
    xmin, ymin, zmin = np.min(kp3d, axis=1) #[np.min(kp3d[:, x]) for x in range(3)]
    xmax, ymax, zmax = np.max(kp3d, axis=1) #[np.max(kp3d[:, x]) for x in range(3)]
    corners = [
        [xmin, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmin],
        [xmin, ymax, zmax],
        [xmax, ymin, zmin],
        [xmax, ymin, zmax],
        [xmax, ymax, zmin],
        [xmax, ymax, zmax]
    ]
    corners = np.array(corners)
    return corners


def norm_kp_in_box(keypoints3d, corners):
    keypoints3d = keypoints3d - corners[0]
    #keypoints3d = keypoints3d / (corners[-1] - corners[0])
    return keypoints3d


def calc_group_bbox(kp3d):
    bbox = []
    x_offset = 100 - 15
    y_offset = -1
    for i in range(1):
        k = kp3d.reshape(14, 3)
        min_x, min_y = np.min(k[:, 0]), np.min(k[:, 1])
        max_x, max_y = np.max(k[:, 0]), np.max(k[:, 1])
        box = (int(min_x / 0.05) + x_offset, 160 - int(max_y / 0.05) + y_offset,
                int(max_x / 0.05) + x_offset, 160 - int(min_y / 0.05) + y_offset)
        pad = 0
        box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        #center = (int(k[0, 0] / 0.05) + x_offset, 160 - int(k[0, 1] / 0.05) + y_offset)
        #bbox_centers.append(center)
        bbox.append(box)
    print(kp3d)
    print(bbox)
    return bbox


def crop_min_mask(mask_dir, kp_dir, kp_out=None, box_out=None):
    save = False if kp_out is None or box_out is None else True
    if kp_out and not os.path.exists(kp_out):
        os.makedirs(kp_out)
    if box_out and not os.path.exists(box_out):
        os.makedirs(box_out)
    for i in range(590):
        file_name = 'png_%04d.npy' % i
        file_name = os.path.join(mask_dir, file_name)
        if not os.path.exists(file_name):
            continue
        mask = np.load(file_name)
        kp_name = '%04d.npy' % i
        kp_name = os.path.join(kp_dir, kp_name)
        if not os.path.exists(kp_name):
            #print('???')
            continue
        keypoints3d = np.load(kp_name)
        #calc_group_bbox(keypoints3d)
        keypoints2d, kp_box = project2img(keypoints3d, cam_index=6, pad=25)
        #print(keypoints3d)
        #print(keypoints2d)
        #print(kp_box)
        #exit()
        kp_box = np.array(kp_box)
        kp2d_off = keypoints2d - kp_box[:2]
        kp2d_norm = kp2d_off / (kp_box[2:] - kp_box[:2])
        #corners = bbox3d(keypoints3d)
        #print(corners)
        #canvas = draw_points2d(keypoints2d)
        #canvas = cv2.rectangle(canvas, tuple(kp_box[:2]), tuple(kp_box[2:]), (255, 255, 255), 2)
        #canvas = draw_points2d(keypoints2d, canvas)

        #plt.imshow(canvas)
        #plt.pause(10)
        #plt.clf()
        #exit()
        #norm_kp = norm_kp_in_box(keypoints3d, corners)
        #print(norm_kp)
        #np.save('norm_test.npy', norm_kp)
        #exit()
        #corners, box = project2img(corners, cam_index=0, pad=0)
        #norm_kp2d, box = project2img(norm_kp, cam_index=0)
        '''
        mask = np.stack([mask * 255] * 3, axis=-1)
        canvas = draw_points2d(keypoints2d, canvas=mask)
        canvas = draw_points2d(kp2d_off, canvas=canvas)
        cv2.rectangle(canvas, tuple(kp_box[:2]), tuple(kp_box[2:]), (255, 255, 255), 4)
        plt.imshow(canvas)
        plt.pause(0.1)
        plt.clf()
        #exit()
        '''
        if save:
            np.save(os.path.join(kp_out, '%04d.npy' % i), kp2d_norm)
            np.save(os.path.join(box_out, '%04d.npy' % i), kp_box)


if __name__ == '__main__':
    prefix = '/data/TIPoseData/view10'
    _ = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28, 29, 30, 31, 36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    for i in _:
        mask_dir = os.path.join(prefix, 'video', '%04d' % i)
        kp_dir = os.path.join(prefix, 'keypoint', '%04d' % i)
        kp_out = os.path.join(prefix, 'proj_kp2d', '%04d' % i)
        box_out = os.path.join(prefix, 'mask_box', '%04d' % i)
        #kp_out, box_out = None, None
        crop_min_mask(mask_dir, kp_dir, kp_out=kp_out, box_out=box_out)
        print(i)
    
