'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-07-20 19:33:01
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-07-20 19:39:42
 # @ Description: Collection of useful tool functions.
 '''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle, Circle
import matplotlib.lines as lines


def draw_points2d(kp2d, shape=(1248, 1640, 3), canvas = None):
    """Visualize 2d keypoints.

    Args:
        kp2d (np.ndarray): 2D keypoints with shape (N, 2).
        shape (tuple(int, int), optional): The shape of background image. Defaults to (1248, 1640, 3).
        canvas (np.ndarray, optional): Background image. Defaults to None.

    Returns:
        np.ndarray: Images containing keypoints.
    """
    if canvas is None:
        canvas = np.zeros(shape, dtype=np.uint8)
    for k in kp2d:
        x, y = k
        center = (int(x), int(y))
        cv2.circle(canvas, center, 3, (255, 255, 255), 2)
    return canvas

def draw_keypoints(p_pr, line_pr, joints):
    """Draw 3D keypoints.

    Args:
        p_pr (np.ndarray): Human 3D keypoints.
        line_pr (list): List of line2d object.
        joints (list): Keypoint relationship, each element denote parent keypoint index of current keypoint index.
    """
    if p_pr.size == 0:
        return
    for n, (i, j) in enumerate(joints):
        line_pr[n].set_color('g')
        line_pr[n].set_data_3d([p_pr[i, 0], p_pr[j, 0]], [p_pr[i, 1], p_pr[j, 1]], [p_pr[i, 2], p_pr[j, 2]])

def init_3d_plot(ax, num_people):
    """Initialize axes.

    Args:
        ax (Axes): Axes object in matplotlib.
        num_people (int): Number of persons.

    Returns:
        _type_: _description_
    """
    ax.set_xlim3d([-2, 2])
    # ax.set_xlabel('X')
    ax.set_ylim3d([2, 6])
    # ax.set_ylabel('Y')
    ax.set_zlim3d([-8, -4])
    # ax.set_zlabel('Z')

    ax.set_title('3DPose', fontsize=10)
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

def visualize(data_item, out=None):
    """Visualize data item.

    Args:
        data_item (tuple): Data sample from HIBER Dataset.
        out (str, optional): Output image filename. Defaults to None.
    """
    plt.ion()
    hor, ver, pose2d, pose3d, hbox, vbox, silhouette = data_item
    figure = plt.figure(figsize=(10, 2.5))
    axes = figure.subplots(1, 4)
    # figure.tight_layout()
    figure.subplots_adjust(0.01, -0.10, 0.99, 0.99, 0, 0)

    # If RF frame is not channel_first, transpose it to channel_first
    if hor.shape[-2] == 2:
        hor, ver = hor.transpose((2, 0, 1)), ver.transpose((2, 0, 1))
        hor, ver = np.ascontiguousarray(hor), np.ascontiguousarray(ver)

    # If RF frame is complex, use its real value when visualize.
    if hor.dtype == np.complex128:
        hor, ver = np.abs(hor), np.abs(ver)
        
    hor, ver = hor[0], ver[0]

    axes[0].imshow(hor)
    axes[0].set_title('Horizontal Heatmap', pad=19.5, fontsize=10)
    axes[1].imshow(ver)
    axes[1].set_title('Vertical Heatmap', pad=19.5, fontsize=10)
    axes[0].set_axis_off()
    axes[1].set_axis_off()

    for box in hbox.astype(np.int):
        x0, y0, x1, y1 = box
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='g', facecolor='none')
        axes[0].add_patch(rect)

    for box in vbox.astype(np.int):
        x0, y0, x1, y1 = box
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='g', facecolor='none')
        axes[1].add_patch(rect)

    silhouette = np.max(silhouette, axis=0) if silhouette.shape[0] != 0 else np.zeros((1248, 1640), dtype=bool)
    axes[2].imshow(silhouette)
    axes[2].set_title('Silhouette & 2D Pose', pad=23, fontsize=10)
    joints = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [1, 8], [1, 11]]
    
    for kp in pose2d.reshape(-1, 14, 2):
        for jj in joints:
            xs, ys = [kp[jj[0], 0], kp[jj[1], 0]], [kp[jj[0], 1], kp[jj[1], 1]]
            axes[2].add_artist(lines.Line2D(xs, ys, color='b'))
    axes[2].set_axis_off()

    axes[3].remove()
    ax3d = figure.add_subplot(1, 4, 4, projection='3d')
    line_pr, joints = init_3d_plot(ax3d, pose3d.shape[0])
    pose3d = pose3d[:, :, :3].reshape(-1, 3)
    draw_keypoints(pose3d, line_pr, joints)
    ax3d.xaxis.set_ticklabels([])
    ax3d.yaxis.set_ticklabels([])
    ax3d.zaxis.set_ticklabels([])

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.draw()
    if out is None:
        plt.show()
        plt.pause(10)
    else:
        plt.savefig(out)
    plt.close(figure)
    plt.ioff()