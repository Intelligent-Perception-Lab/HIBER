from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm


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

def build_multi_camera_system_from_dict(camera_names, camera_params):
    '''
    bulid multi-camera system form camera params dict
    :param camera_names: list contains cameras to bulid system
    :param camera_params: dict contains all cameras' params
    :return: camera system
    '''
    pymvg_cameras = []
    for cam in camera_names:
        if not cam in camera_params.keys():
            continue
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

def single_cam_vec(cam, axes_size=0.2):
    C = cam.get_camcenter()
    world_coords = cam.project_camera_frame_to_3d([[axes_size,0,0],
                                                   [0,axes_size,0],
                                                   [0,0,axes_size]])
    vv = world_coords[2]
    v = np.vstack(([C], [vv]))
    v_roll = np.roll(v, -1, axis=0)
    _ = v_roll - v
    return _[0, :]

def cam_vec(cam):
    if isinstance(cam, list):
        res = []
        for ca in cam:
            res.append(single_cam_vec(ca))
        return res
    else:
        return single_cam_vec(cam)

def build_camera_from_dict(camera, cam_name):
    W, H, R, T, K, D, P = unfold_camera_param_from_dict(camera[cam_name])
    camera = CameraModel.load_camera_from_M(
                P, width=W, height=H, name=cam_name, distortion_coefficients=D)
    return camera

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

def rotation_matrix(cam_index=0, cam_file='view1_cam2_cal.json'):
    #cam_file = 'view1_cam2_cal.json'
    with open(cam_file) as f:
        cam_data = json.load(f)

    cam0 = build_camera_from_dict(cam_data, 'cam_0')
    camt = build_camera_from_dict(cam_data, 'cam_%d' % cam_index)
    vec0, vect = cam_vec([cam0, camt])
    vect[-1] = 0
    r_mat = rotation_matrix_from_vectors(vec0, vect)
    return r_mat

def calc_rmats():
    cam_files = glob.glob('/mnt/hdd/data/view*/view*_cam*_cal.json')
    cam_files = sorted(cam_files)
    cam_files.append(cam_files.pop(1))
    # cam_file = cam_files[view - 1]
    cams = []
    cam0s = []
    for cam_file in cam_files:
        cam_index = int(cam_file.split('_')[1][3:])
        with open(cam_file) as f:
            cam_data = json.load(f)
        cam_names = ['cam_%d' % x for x in range(13)]
        cam_system = build_multi_camera_system_from_dict(cam_names, cam_data)        
        cam = cam_system._cameras[cam_names[cam_index]]
        cam0 = build_camera_from_dict(cam_data, 'cam_0')
        cams.append(cam)
        cam0s.append(cam0)
    rotation_matrixes = []
    for c0, c in zip(cam0s, cams):
        vec0, vect = cam_vec([c0, c])
        vect[-1] = 0
        rotation_matrixes.append(rotation_matrix_from_vectors(vec0, vect))
    return rotation_matrixes

def load_offs():
    off_files = glob.glob('/mnt/hdd/data/view*/offset.npy')
    off_files = sorted(off_files)
    off_files.append(off_files.pop(1))
    offsets = []
    for off_file in off_files:
        offset = np.load(off_file)
        offsets.append(offset)
    return offsets

def compensate(x, y, r_off):
    y_ = (x - r_off) ** 2 + (160 - y) ** 2
    y_ = y_ ** 0.5
    compen = y_ - (160 - y)
    return compen

def apply(keypoints, offset):
    return keypoints + offset[:3]

def draw_keypoints(p_pr, line_pr, joints):
    for n, (i, j) in enumerate(joints):
        line_pr[n].set_color('g')
        line_pr[n].set_data_3d([p_pr[i, 0], p_pr[j, 0]], [p_pr[i, 1], p_pr[j, 1]], [p_pr[i, 2], p_pr[j, 2]])

def init_3d_plot(ax, num_people):
    # ax.set_xlim3d([-5.5, 7.5])
    # # ax.set_xlabel('X')
    # ax.set_ylim3d([2.0, 5.0])
    # # ax.set_ylabel('Y')
    # ax.set_zlim3d([-2.0, 1.0])
    ax.set_xlim3d([-45, 45])
    # ax.set_xlabel('X')
    ax.set_ylim3d([35, 135])
    # ax.set_ylabel('Y')
    ax.set_zlim3d([-120, -100])
    # ax.set_zlabel('Z')
    ax.set_title('Pose Animation', fontsize=10)
    joints = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [1, 8], [1, 11]] #, [14, 15], [15, 16], [16, 17], [14, 17]]
    j_len = len(joints)
    for i in range(j_len, j_len * num_people):
        _ = joints[i % j_len]
        offset = i // j_len * 14
        _ = [x + offset for x in _]
        joints.append(_)
    line_pr = []
    for _ in range(len(joints)):
        line_pr.append(ax.plot([], [], [])[0])
    return line_pr, joints

# kp_jsons = '/mnt/hdd/data/view*/keypoint/train_view*_*.json'
r_mats = calc_rmats()
offsets = load_offs()
# kp_jsons = glob.glob(kp_jsons)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
line_pr, joints = init_3d_plot(ax, 1)
kps = '/mnt/hdd/hiber/*/ANNOTATIONS/3DPOSE/*.npy'
kps = glob.glob(kps)
kps = sorted(kps)
# kps = ['/mnt/hdd/hiber/MULTI/ANNOTATIONS/3DPOSE/03_59.npy', 
#         '/mnt/hdd/hiber/MULTI/ANNOTATIONS/3DPOSE/03_60.npy', 
#         '/mnt/hdd/hiber/MULTI/ANNOTATIONS/3DPOSE/03_61.npy']
for kp_path in tqdm(kps):
    v, g = os.path.basename(kp_path)[:-4].split('_')
    v, g = int(v), int(g)
    kp = np.load(kp_path)
    seq, num_p, num_kp, coor = kp.shape

    kp = kp.reshape(-1, 3)
    kp = kp @ r_mats[v - 1]
    kp = kp / 0.05
    kp[:, [1, 2]] = -kp[:, [1, 2]]
    kp = apply(kp, offsets[v - 1])
    kp[:, [2]] = -kp[:, [2]]
    kp[:, [0, 1]] = kp[:, [0, 1]] - np.array([100, 160])
    kp[:, [1]] = -kp[:, [1]]
    kp = kp * 0.05
    kp = kp.reshape(seq, num_p, num_kp, coor)
    np.save(kp_path, kp)
    # kp = kp.reshape(-1, num_kp, coor)
    # for k in kp:
    #     draw_keypoints(k, line_pr, joints)
    #     plt.draw()
    #     # plt.pause(10)
    #     #assert False
    #     plt.savefig('test3d.jpg')
