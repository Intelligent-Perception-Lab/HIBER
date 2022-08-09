# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fftpack import rfft, irfft, fftfreq, fft


# # Number of samplepoints
# N = 500
# # sample spacing
# T = 0.1

# x = np.linspace(0.0, (N-1)*T, N)
# # x = np.arange(0.0, N*T, T)  # alternate way to define x
# y = 5*np.sin(x) + np.cos(2*np.pi*x) 

# yf = fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# #fft end

# f_signal = rfft(y)
# W = fftfreq(y.size, d=x[1]-x[0])
# plt.plot(W)
# plt.savefig('temp.jpg')
# plt.clf()
# cut_f_signal = f_signal.copy()
# cut_f_signal[(W>0.6)] = 0  # filter all frequencies above 0.6

# cut_signal = irfft(cut_f_signal)

# # plot results
# f, axarr = plt.subplots(1, 3, figsize=(9, 3))
# axarr[0].plot(x, y)
# axarr[0].plot(x,5*np.sin(x),'g')

# axarr[1].plot(xf, 2.0/N * np.abs(yf[:N//2]))
# axarr[1].legend(('numpy fft * dt'), loc='upper right')
# axarr[1].set_xlabel("f")
# axarr[1].set_ylabel("amplitude")


# axarr[2].plot(x,cut_signal)
# axarr[2].plot(x,5*np.sin(x),'g')

# plt.savefig('temp.jpg')

def fit(x):
    z1 = np.polyfit(np.arange(0, len(x)), x, 1)
    p1 = np.poly1d(z1)
    return p1


def calc_std(kps):
    # kps: 590, num_people, 14, 3
    diff = kps - np.median(kps, axis=2, keepdims=True)
    return np.abs(diff)
    # diff = np.sum(np.abs(diff), axis=-1)
    # # stds = np.max(stds, axis=-1)
    # # stds = np.max(stds, axis=0)
    # # kps = kps.copy().transpose(1, 0, 2, 3).reshape(-1, 590 * 14, 3)
    # # stds = np.std(kps, axis=1)
    # # stds = np.mean(stds, axis=1)
    # # point_stds = np.std(kps, )
    return diff

import numpy as np
import matplotlib.pyplot as plt
keypoints = np.load('/mnt/hdd/hiber/ACTION/ANNOTATIONS/3DPOSE/03_49.npy')
diffs = calc_std(keypoints)
for kp_seq in range(keypoints.shape[-2]):
    pts = keypoints[:, 0, kp_seq, :]
    diff = diffs[:, 0, kp_seq, :]
    pts
points = keypoints[:, 0, 4, 0]
plt.plot(points)
plt.savefig('frame.jpg')
points