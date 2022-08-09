import os
import numpy as np
import json
import matplotlib.pyplot as plt
import glob



def draw_keypoints(p_pr, line_pr, joints):
    for n, (i, j) in enumerate(joints):
        line_pr[n].set_color('g')
        line_pr[n].set_data_3d([p_pr[i, 0], p_pr[j, 0]], [p_pr[i, 1], p_pr[j, 1]], [p_pr[i, 2], p_pr[j, 2]])

def init_3d_plot(ax, num_people):
    ax.set_xlim3d([-5.5, 7.5])
    # ax.set_xlabel('X')
    ax.set_ylim3d([-5.0, -3.0])
    # ax.set_ylabel('Y')
    ax.set_zlim3d([-6.0, -4.0])
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

figure = plt.figure()
ax = figure.add_subplot(1, 1, 1, projection='3d')
line_pr, joints = init_3d_plot(ax, 1)

prefix = '/mnt/hdd/hiber/temp/corrected_Annotations'
keypoints = glob.glob(os.path.join(prefix, '*.json'))
keypoints = sorted(keypoints)

for kp_file in keypoints:
    with open(kp_file) as f:
        kp_data = json.load(f)
    keypoints = kp_data['annotations']
    for keypoint in keypoints:
        kp = keypoint['keypoints']
        kp = np.array(kp).reshape(-1, 4)
        draw_keypoints(kp, line_pr, joints)
        plt.draw()
        plt.savefig('corrected.jpg')
