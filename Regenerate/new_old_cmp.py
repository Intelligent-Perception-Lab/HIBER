import matplotlib.pyplot as plt
import json
import numpy as np
import os

def draw_keypoints(p_pr, line_pr, joints):
    for n, (i, j) in enumerate(joints):
        line_pr[n].set_color('g')
        line_pr[n].set_data_3d([p_pr[i, 0], p_pr[j, 0]], [p_pr[i, 1], p_pr[j, 1]], [p_pr[i, 2], p_pr[j, 2]])

def init_3d_plot(ax, num_people, name='Pose Animation'):
    ax.set_xlim3d([-3.5, 3.5])
    ax.set_xlabel('X')
    #ax.set_ylim3d([0.0, 7.0])
    ax.set_ylim3d([0.0, 7.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-2.0, 0.2])
    ax.set_zlabel('Z')
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

# def init_figure(num_people):
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     #ax.set_xlim3d([-3.5, 3.5])
#     ax.set_xlim3d([-3.5, 3.5])
#     ax.set_xlabel('X')
#     #ax.set_ylim3d([0.0, 7.0])
#     ax.set_ylim3d([0.0, 7.0])
#     ax.set_ylabel('Y')
#     ax.set_zlim3d([-2.0, 0.2])
#     ax.set_zlabel('Z')
#     #ax.set_axis_off()
#     #plt.xticks([])
#     #plt.yticks([])
#     #plt.zticks([])
#     #ax.set_title('Pose Animation')
#     joints = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [1, 8], [1, 11]] #, [14, 15], [15, 16], [16, 17], [14, 17]]
#     j_len = len(joints)
#     for i in range(j_len, j_len * num_people):
#         _ = joints[i % j_len]
#         offset = i // j_len * 14
#         _ = [x + offset for x in _]
#         joints.append(_)
#     line_pr = []
#     for _ in range(len(joints)):
#         line_pr.append(ax.plot([], [], [])[0])
#     plt.ion()
#     return line_pr, joints

def tranform_k(k):
    k = np.array(k) / 2
    k = k.reshape(-1, 3)
    k[:, 0] = k[:, 0] + 3
    k[:, 1] = k[:, 1] - 5
    return k


if __name__ == "__main__":
    # prefix = 'NewAnno/no_smooth_results'
    # filename = 'train_view9_48_multi_new.json'
    # filename = 'train_view3_61_multi_new.json'
    # filename = os.path.join(prefix, filename)
    # num_people = 1 if -1 != filename.find('multi') else 1

    # line_pr, joints = init_figure(num_people)

    # with open(filename) as f:
    #     a = json.load(f)

    # b = a['annotations']
    # for c in b:
    #     c = c['keypoints']
    #     c = np.array(c) / 2
    #     c = c.reshape(-1, 3)
    #     c[:, 0] = c[:, 0] + 3
    #     c[:, 1] = c[:, 1] - 5

    #     draw_keypoints(c)
    #     plt.pause(0.001)
    # plt.ioff()
    old = '/home/virgil/Documents/HIBER/Regenerate/OldAnno'
    new = '/home/virgil/Documents/HIBER/Regenerate/NewAnno/no_smooth_results'
    pattern = 'train_view3_61_multi_new.json'
    
    old_kp = os.path.join(old, pattern)
    new_kp = os.path.join(new, pattern)

    fig = plt.figure()
    ax_old = fig.add_subplot(121, projection='3d')
    ax_new = fig.add_subplot(122, projection='3d')

    l0, j0 = init_3d_plot(ax_old, 2, name='Old Pose')
    l1, j1 = init_3d_plot(ax_new, 2, name="New Pose")

    with open(old_kp) as f:
        o_kp = json.load(f)['annotations']
    with open(new_kp) as f:
        n_kp = json.load(f)['annotations']

    for o, n in zip(o_kp, n_kp):
        o, n = o['keypoints'], n['keypoints']
        o = tranform_k(o)
        draw_keypoints(o, l0, j0)
        n = tranform_k(n)
        draw_keypoints(n, l1, j1)
        # plt.draw()
        # plt.show()
        # plt.pause(10)
        plt.savefig('cmp.png')

    
