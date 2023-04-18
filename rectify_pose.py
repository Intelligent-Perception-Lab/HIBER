import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import HIBERTools as hiber
from scipy import signal

import scipy.fftpack

def denoise(y):
    # f, t, Zxx = signal.stft(y, fs=1, nperseg=2)

    # x = np.linspace(1, len(y), len(y))
    w = scipy.fftpack.rfft(y)
    # f = scipy.fftpack.rfftfreq(len(x), x[1]-x[0])
    spectrum = w**2
    plt.plot(spectrum)
    plt.savefig('rectify.jpg')

    # cutoff_idx = spectrum > 50
    w2 = w.copy()
    # w2[cutoff_idx] = 0
    y2 = scipy.fftpack.irfft(w2)
    
    plt.clf()
    plt.plot(y2)
    plt.savefig('rectify.jpg')
    return y2

def denoise3d(pose):
    plt.plot(pose[:, 0, 0, 0])
    plt.savefig('rectify.jpg')
    w = scipy.fftpack.rfft(pose, axis=0)
    spectrum = w**2
    # plt.plot(spectrum[:, 0, 0, 0])
    # plt.savefig('rectify.jpg')
    cutoff_idx = spectrum < 2e2
    w2 = w.copy()
    w2[cutoff_idx] = 0
    y2 = scipy.fftpack.irfft(w2)
    plt.clf()
    plt.plot(y2[:, 0, 0, 0])
    plt.savefig('rectify.jpg')
    return y2


def draw_keypoints(p_pr, line_pr, joints):
    for n, (i, j) in enumerate(joints):
        if n < 13:
            line_pr[n].set_color('g')
        else:
            line_pr[n].set_color('b')
        line_pr[n].set_data_3d([p_pr[i, 0], p_pr[j, 0]], [p_pr[i, 1], p_pr[j, 1]], [p_pr[i, 2], p_pr[j, 2]])

def init_3d_plot(ax, num_people, name='3D Pose'):
    # ax.set_xlim3d([-2, 2])
    # # ax.set_xlabel('X')
    # ax.set_ylim3d([-13, -11])
    # # ax.set_ylabel('Y')
    # ax.set_zlim3d([-8, -4])
    # # ax.set_zlabel('Z')

    ax.set_xlim3d([-2, 2])
    # ax.set_xlabel('X')
    ax.set_ylim3d([2, 6])
    # ax.set_ylabel('Y')
    ax.set_zlim3d([-8, -4])
    # ax.set_zlabel('Z')
    ax.set_title(name, fontsize=10)
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

def corrected_annos():
    cors = []
    corrected = '/mnt/hdd/hiber/temp/corrected_Annotations/*.json'
    corrected = glob.glob(corrected)
    for cor in corrected:
        basename = os.path.basename(cor).split('_')[-1].split('.')[0]
        v, g = int(basename[:2]), int(basename[2:])
        cors.append('%02d_%02d' % (v, g))
    return cors

def calc_std(kps):
    # kps: 590, num_people, 14, 3
    stds = np.std(kps, axis=2)
    stds = np.max(stds, axis=-1)
    stds = np.max(stds, axis=0)
    # kps = kps.copy().transpose(1, 0, 2, 3).reshape(-1, 590 * 14, 3)
    # stds = np.std(kps, axis=1)
    # stds = np.mean(stds, axis=1)
    # point_stds = np.std(kps, )
    return stds

# def reorder(kps):
#     # kps 590, 2, 14, 3

#     pass


pose3d = '/mnt/hdd/hiber/MULTI/ANNOTATIONS/3DPOSE/*.npy'
pose3ds = glob.glob(pose3d)

corrected = corrected_annos()

plt.ion()
kp_stds = []
for p3d in pose3ds:
    v, g = p3d.split('/')[-1].split('_')[0], p3d.split('/')[-1].split('_')[1][:2]
    v, g = int(v), int(g)
    if '%02d_%02d' % (v, g) in corrected:
        continue
    kps = np.load(p3d)
    # new_kps = reorder(kps)
    # kp_std = calc_std(kps)
    # kp_stds.append(kp_std)
    # if kp_std > 1:
        # denoise(kps[:, 0, 4, 0])
        # denoise3d(kps)
        # assert False
    figure = plt.figure()
    figure.suptitle('%02d_%02d' % (v, g))
    ax0 = figure.add_subplot(1, 2, 1, projection='3d')
    ax1 = figure.add_subplot(1, 2, 2, projection='3d')
    l0, j0 = init_3d_plot(ax0, kps.shape[1], name='old')
    l1, j1 = init_3d_plot(ax1, kps.shape[1], name='new')
    for i, kp in enumerate(kps):
        figure.suptitle('%02d_%02d_%04d' % (v, g, i))
        # if i > 20:
        #     break
        draw_keypoints(kp.reshape(-1, 3), l0, j0)
        draw_keypoints(kp.reshape(-1, 3), l1, j1)
        plt.draw()
        plt.show()
        plt.pause(0.01)
        # plt.savefig('frame.jpg')
        # plt.close(figure)
        # ax0.cla()
        # ax1.cla()
    plt.pause(20)
    plt.close(figure)
# print(kp_stds)
