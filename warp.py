

import numpy as np
import cv2
from scipy import interpolate

def warpH(im2, H2to1, outsize, LAB_space=False, kind='linear'):
    if (LAB_space):
        im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(im2)

    x_im2 = np.arange(l_channel.shape[1])

    y_im2 = np.arange(l_channel.shape[0])

    x = []
    y = []
    z = []
    for i in range(outsize[0]):
        for j in range(outsize[1]):
            x.append(j)
            y.append(i)
            z.append(1)

    p_old = np.array([x, y, z])
    p_new_temp = np.linalg.inv(H2to1) @ p_old
    p_new = np.array([p_new_temp[0, :] / p_new_temp[2, :], p_new_temp[1, :] / p_new_temp[2, :]])

    f_l = interpolate.interp2d(x_im2, y_im2, l_channel, kind=kind)
    f_a = interpolate.interp2d(x_im2, y_im2, a_channel, kind=kind)
    f_b = interpolate.interp2d(x_im2, y_im2, b_channel, kind=kind)

    znew_l = []
    znew_a = []
    znew_b = []
    for i in range(p_old.shape[1]):
        if (p_new[0, i] > 0 and p_new[1, i] > 0 and p_new[0, i] < im2.shape[1] and p_new[1, i] < im2.shape[0]):
            znew_l_temp = np.round((f_l(p_new[0, i], p_new[1, i])))
            znew_l_temp = (znew_l_temp[0]).astype('uint8')
            znew_a_temp = np.round((f_a(p_new[0, i], p_new[1, i])))
            znew_a_temp = (znew_a_temp[0]).astype('uint8')
            znew_b_temp = np.round((f_b(p_new[0, i], p_new[1, i])))
            znew_b_temp = (znew_b_temp[0]).astype('uint8')
        else:
            znew_l_temp = 0
            znew_a_temp = 0
            znew_b_temp = 0
        znew_l.append(znew_l_temp)
        znew_a.append(znew_a_temp)
        znew_b.append(znew_b_temp)

    znew_l = (np.array(znew_l).reshape((outsize[0], outsize[1]))).astype('uint8')
    znew_a = (np.array(znew_a).reshape((outsize[0], outsize[1]))).astype('uint8')
    znew_b = (np.array(znew_b).reshape((outsize[0], outsize[1]))).astype('uint8')

    warp_im2 = np.stack([znew_l, znew_a, znew_b], 2)

    if (LAB_space):
        warp_im2 = cv2.cvtColor(warp_im2, cv2.COLOR_LAB2RGB)


    return warp_im2