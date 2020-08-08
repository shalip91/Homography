import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt

#Add imports if needed:
from scipy import interpolate
import time
#end imports

#Add extra functions here:
def creatPanom(HpanoList, outsize, filepath='sintra/sintra'):
    """""stiching"""""
    """1to2"""
    img = cv2.imread(filepath + '1_pyr.png')
    imgB = cv2.imread(filepath + '2_pyr.png')
    warpImg1to2 = warpH(img, HpanoList[0], outsize, LAB_space=False, kind='linear')
    pano12 = imageStitching(imgB, warpImg1to2)

    """12to3"""
    img = pano12
    imgB = cv2.imread(filepath + '3_pyr.png')
    warpImg12to3 = warpH(img, HpanoList[1], outsize, LAB_space=False, kind='linear')
    pano123 = imageStitching(imgB, warpImg12to3)

    """123to4"""
    imgB = cv2.imread(filepath + '4_pyr.png')
    img = pano123
    warpImg123to4 = warpH(img, HpanoList[2], outsize, LAB_space=False, kind='linear')
    pano1234 = imageStitching(imgB,warpImg123to4)

    """5to1234"""
    img = cv2.imread(filepath + '5_pyr.png')
    imgB = pano1234
    warpImg5to1234 = warpH(img, HpanoList[4], outsize, LAB_space=False, kind='linear')
    pano12345 = imageStitching(imgB,warpImg5to1234)
    pano12345 = whiteBackground(pano12345)
    cv2.imwrite(filepath + "_pano.png", pano12345)

    return pano12345

def creatHtoImg4(index=[1,2,3,4,5], target=4, N=6,  SiftTreshhold=0.3,
                  filepath='sintra/sintra',manual=False, RANSAC=False, nIter=1000, tol=1):

    HpanoList = []
    for i in index:
        if (i == target):
            HpanoList.append(0)
        if (i<target):
            filepath_projection = filepath + str(i) + '_pyr.png'
            filepath_base = filepath + str(i+1) + '_pyr.png'
            img = cv2.imread(filepath_projection)
            imgB = cv2.imread(filepath_base)
            if (manual):
                p1, p2 = getPoints(cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
            else:
                p1, p2 = getPoints_SIFT(imgB, img, N=N, treshhold=SiftTreshhold)

            if(RANSAC):
                H_sift_sintra = ransacH(p1, p2, nIter=nIter, tol=tol)
            else:
                H_sift_sintra = computeH(p1, p2)
            HpanoList.append(H_sift_sintra)
        if (i>target):
            filepath_projection = filepath + str(i) + '_pyr.png'
            filepath_base = filepath + str(i-1) + '_pyr.png'
            img = cv2.imread(filepath_projection)
            imgB = cv2.imread(filepath_base)
            if (manual):
                p1, p2 = getPoints(cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
            else:
                p1, p2 = getPoints_SIFT(imgB, img, N=N, treshhold=SiftTreshhold)

            if (RANSAC):
                H_sift_sintra = ransacH(p1, p2, nIter=nIter, tol=tol)
            else:
                H_sift_sintra = computeH(p1, p2)
            HpanoList.append(H_sift_sintra)

    return HpanoList

def whiteBackground(img):

    white = np.ones((img.shape[0], img.shape[1], 3), np.uint8)*255

    for y in range(white.shape[0]):
        for x in range(white.shape[1]):
            if (img[y, x, 0] > 0):
                white[y, x, :] = img[y, x, :]

    return white

def pyrDownImages(pyrDownIter=2, filepath='sintra/sintra'):
    img1 = cv2.imread(filepath + '1.JPG')
    img2 = cv2.imread(filepath + '2.JPG')
    img3 = cv2.imread(filepath + '3.JPG')
    img4 = cv2.imread(filepath + '4.JPG')
    img5 = cv2.imread(filepath + '5.JPG')

    while (pyrDownIter > 0):
        pyrDownIter = pyrDownIter - 1
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)
        img3 = cv2.pyrDown(img3)
        img4 = cv2.pyrDown(img4)
        img5 = cv2.pyrDown(img5)

    cv2.imwrite(filepath + '1_pyr.png', img1)
    cv2.imwrite(filepath + '2_pyr.png', img2)
    cv2.imwrite(filepath + '3_pyr.png', img3)
    cv2.imwrite(filepath + '4_pyr_real.png', img4)
    cv2.imwrite(filepath + '5_pyr.png', img5)

def creatHtoImgPano(index=[1,2,3,4,5], target=4, filepath='sintra/sintra'):

    Hlist = []
    HpanoList = []
    for i in index:
        if (i<target):
            filepath_projection = filepath + str(i) + '_pyr.png'
            filepath_base = filepath + str(i+1) + '_pyr.png'
            img = cv2.imread(filepath_projection)
            imgB = cv2.imread(filepath_base)
            p1, p2 = getPoints_SIFT(imgB, img, 15, treshhold=0.15)
            H_sift_sintra = computeH(p1, p2)
            Hlist.append(H_sift_sintra)
        if (i>target):
            filepath_projection = filepath + str(i) + '_pyr.png'
            filepath_base = filepath + str(i-1) + '_pyr.png'
            img = cv2.imread(filepath_projection)
            imgB = cv2.imread(filepath_base)
            p1, p2 = getPoints_SIFT(imgB, img, 15, treshhold=0.15)
            H_sift_sintra = computeH(p1, p2)
            Hlist.append(H_sift_sintra)

    HpanoList.append(np.matmul(Hlist[0], Hlist[1]))
    HpanoList.append(Hlist[1])
    HpanoList.append(0)
    HpanoList.append(Hlist[2])
    HpanoList.append(Hlist[3] @ Hlist[2])

    return HpanoList

def baseImageTranslation(img, outsize, shiftX, shiftY):

    ImageTran = np.zeros((outsize[0], outsize[1], 3), np.uint8)

    for y in range(ImageTran.shape[0]):
        for x in range(ImageTran.shape[1]):
            if (x < img.shape[1] and y < img.shape[0]):
                ImageTran[y + shiftY, x + shiftX, :] = img[y, x, :]

    return ImageTran

#Extra functions end

# HW functions:
def getPoints(im1,im2,N):
    """remarks N points in img 1 then matching them in img 2"""
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB))
    p1 = np.array(plt.ginput(N,0)).T
    plt.close()
    plt.imshow(cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB))
    p2 = np.array(plt.ginput(N,0)).T
    plt.close()

    return p1,p2

def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    A = np.zeros((2 * p2.shape[1], 9))
    H2to1 = []

    """building matrix A"""
    for i in range(0, 2 * p2.shape[1] - 1, 2):
        A[i][0] = p2[0, i // 2]
        A[i][1] = p2[1, i // 2]
        A[i][2] = 1

        A[i + 1][3] = p2[0, i // 2]
        A[i + 1][4] = p2[1, i // 2]
        A[i + 1][5] = 1

        A[i][6] = -p2[0, i // 2] * p1[0, i // 2]
        A[i][7] = -p2[1, i // 2] * p1[0, i // 2]
        A[i][8] = -p1[0, i // 2]

        A[i + 1][6] = -p2[0, i // 2] * p1[1, i // 2]
        A[i + 1][7] = -p2[1, i // 2] * p1[1, i // 2]
        A[i + 1][8] = -p1[1, i // 2]

    D, V = np.linalg.eig(A.T @ A)
    H2to1 = np.array(V[:, -1]).reshape((3,3))

    return H2to1

def computeAffineH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    A = np.zeros((2 * p2.shape[1], 6))
    b = np.zeros((2 * p2.shape[1], 1))

    """building matrix A"""
    for i in range(0, 2 * p2.shape[1] - 1, 2):
        A[i][0] = p2[0, i // 2]
        A[i][1] = p2[1, i // 2]
        A[i][2] = 1

        A[i + 1][3] = p2[0, i // 2]
        A[i + 1][4] = p2[1, i // 2]
        A[i + 1][5] = 1



        b[i] = p1[0, i // 2]
        b[i+1] = p1[1, i // 2]

    affinH2to1 = np.array(np.linalg.inv(A.T @ A) @ A.T @ b).reshape((2,3))
    thirdLine = np.array([[0, 0, 1]])
    affinH2to1 = np.concatenate((affinH2to1, thirdLine), axis=0)

    return affinH2to1

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

def imageStitching(img1, wrap_img2):

    panoImg = np.zeros((wrap_img2.shape[0], wrap_img2.shape[1], 3), np.uint8)

    for y in range(panoImg.shape[0]):
        for x in range(panoImg.shape[1]):
            if (x < img1.shape[1] and y < img1.shape[0]):
                panoImg[y, x, :] = img1[y, x, :]
            if (wrap_img2[y, x, 0] > 0):
                panoImg[y, x, :] = wrap_img2[y, x, :]

    return panoImg

def ransacH(p1, p2, nIter, tol):
    p1 = np.array(tuple(zip(p1[0], p1[1])))
    p2 = np.array(tuple(zip(p2[0], p2[1])))
    bestH = cv2.findHomography(p2, p1, method=cv2.RANSAC, ransacReprojThreshold=tol, maxIters=nIter)
    return bestH[0]

def getPoints_SIFT(img1,img2,N=6, treshhold=0.3):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    p1_x = []
    p1_y = []
    p2_x = []
    p2_y = []
    p1 = []
    p2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < treshhold * n.distance and N > 0:
            N = N - 1
            matchesMask[i] = [1, 0]

            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt

            matches_points_1_x = pt1[0]
            p1_x.append(matches_points_1_x)
            matches_points_1_y = pt1[1]
            p1_y.append(matches_points_1_y)

            matches_points_2_x = pt2[0]
            p2_x.append(matches_points_2_x)
            matches_points_2_y = pt2[1]
            p2_y.append(matches_points_2_y)

    p1 = np.array([p1_x, p1_y])
    p2 = np.array([p2_x, p2_y])
    return p1,p2

if __name__ == '__main__':
    print('my_homography')

# # """2.1"""
# im1 = cv2.imread('incline/incline_L_pyr.png',0)
# im2 = cv2.imread('incline/incline_R_pyr.png',0)
# p1, p2 = getPoints(im1, im2, 3)
# print(f"\np1:\n {p1}\n")
# np.save("incline/p2.npy", p2)
#



# # """2.2"""
# H2to1 = computeH(p1, p2)
# print(f"\nH2to1:\n {H2to1}\n")
# np.save("incline/H2to1_pyr.npy", H2to1)
#
#


# """2.3"""
# H2to1 = np.load("incline/H2to1_pyr.npy")
# im2 = cv2.imread('incline/incline_R_pyr.png')
# outsize = np.array([350, 900])
# warp_im2 = warpH(im2, H2to1, outsize, LAB_space=False, kind='cubic')
# cv2.imwrite('incline/warp_im2_cubic.png', warp_im2)
#


#
# """2.4"""
# img1 = cv2.imread('incline/incline_L_pyr.png')
# warp_im2 = cv2.imread('incline/warp_im2.png')
# panoImg = imageStitching(img1, warp_im2)
# cv2.imwrite('incline/panoImg.png', panoImg)
# cv2.imshow("incline/panoImg", panoImg)
# cv2.waitKey(0)
#



"""2.5"""
# img1 = cv2.imread('incline/incline_L_pyr.png')
# img2 = cv2.imread('incline/incline_R_pyr.png')
# p1, p2 = getPoints_SIFT(img1,img2, N=7)
# print("points")
#
# # H2to1_sift = computeH(p1, p2)
#
# print(f"H2to1_sift:\n {H2to1_sift}\n")
# print("H2to1_sift")
#
# outsize = np.array([350, 900])
# warp_im2_sift = warpH(img2, H2to1_sift, outsize, LAB_space=False, kind='cubic')

# cv2.imwrite('incline/warp_im2_cubic_sift.png', warp_im2_sift)
# print("warp_im2_sift")
#
# panoImg_sift = imageStitching(img1, warp_im2_sift)
# cv2.imwrite('incline/panoImg_sift.png', panoImg_sift)
# cv2.imshow("incline/panoImg_sift", panoImg_sift)
# cv2.waitKey(0)




"""2.7"""
"""stiching - SIFT & manual with and without RANSAC for all images"""

# start = time.time()
# filepath = 'sintra/sintra'
# pyrDownIter = 3
# pyrDownImages(pyrDownIter=pyrDownIter, filepath=filepath)
# img4 = cv2.imread(filepath + '4_pyr_real.png')
#
#
# ## hyperParameters for sintra
# if (filepath=='sintra/sintra'):
#     if (pyrDownIter == 0):
#         outsize = np.array([4000, 14400])
#         N = 600
#         SiftTreshhold = 0.12
#     if (pyrDownIter == 1):
#         outsize = np.array([2000, 7200])
#         N = 600
#         SiftTreshhold = 0.13
#     if (pyrDownIter == 2):
#         outsize = np.array([1000, 3600])
#         N = 300
#         SiftTreshhold = 0.14
#     if (pyrDownIter == 3):
#         outsize = np.array([600, 1800])
#         N = 35
#         SiftTreshhold = 0.15
#     img4 = baseImageTranslation(img4, outsize,
#                                 shiftX=(outsize[0] // 2)-(outsize[0] // 9),
#                                 shiftY=(outsize[1] // 7)-(outsize[1] // 17))
#
# ## hyperParameters for beach
# if (filepath=='beach/beach'):
#     if (pyrDownIter == 2):
#         outsize = np.array([2500, 1200])
#         N = 15
#         SiftTreshhold = 0.3
#
#     if (pyrDownIter == 1):
#         outsize = np.array([5000, 2400])
#         N = 50
#         SiftTreshhold = 0.25
#
#     if (pyrDownIter == 0):
#         outsize = np.array([10000, 4800])
#         N = 120
#         SiftTreshhold = 0.14
#     img4 = baseImageTranslation(img4, outsize,
#                                 shiftX=outsize[1] // 5,
#                                 shiftY=outsize[0] // 10)
#
# ## hyperParameters for haifa
# if (filepath=='haifa/haifa'):
#     if (pyrDownIter == 3):
#         SiftTreshhold = 0.15
#         N =100
#         outsize = np.array([900, 2500]) ## beach 2 pyr
#
#     if (pyrDownIter == 2):
#         SiftTreshhold = 0.15
#         N = 100
#         outsize = np.array([1800, 5000])  ## beach 2 pyr
#
#     if (pyrDownIter == 1):
#         SiftTreshhold = 0.15
#         N = 200
#         outsize = np.array([3600, 10000])  ## beach 2 pyr
#     if (pyrDownIter == 0):
#         SiftTreshhold = 0.14
#         N = 400
#         outsize = np.array([7200, 20000])  ## beach 2 pyr
#     img4 = baseImageTranslation(img4, outsize,
#                                 shiftX=outsize[1] // 7 - 100,
#                                 shiftY=outsize[0] // 6)
#
# cv2.imwrite(filepath + '4_pyr.png', img4)
# HpanoList = creatHtoImg4(index=[1,2,3,4,5], target=4, N=N,
#                          SiftTreshhold=SiftTreshhold, filepath=filepath,
#                          manual=False, RANSAC=True, nIter=1000, tol=1)
#
# pano12345 = creatPanom(HpanoList, outsize, filepath=filepath)
# end = time.time()
# print(f"\nRun Time:\n {end-start}\n")
# cv2.imshow(filepath + "_pano.png", pano12345)
#



"""2.11"""
"""affine VS projection"""
"""GOOD RESULTS"""
# img1 = cv2.imread('bay/bay1.jpg')
# img2 = cv2.imread('bay/bay2.jpg')
# outsize = np.array([700, 1500])
# img1 = baseImageTranslation(img1, outsize,
#                                 shiftX=0,
#                                 shiftY=outsize[0] // 10)
# p1, p2 = getPoints_SIFT(img1,img2, N=30, treshhold=0.15)
# print("points")
#
# H2to1_affine = computeAffineH(p1, p2)
# print(f"H2to1_sift:\n {H2to1_affine}\n")
# warp_im2_affine = warpH(img2, H2to1_affine, outsize, LAB_space=False, kind='linear')
# panoImg_affine = imageStitching(img1, warp_im2_affine)
# cv2.imwrite('bay/bay_affine.png', panoImg_affine)
#
# H2to1_projective = computeH(p1, p2)
# print(f"H2to1_sift:\n {H2to1_projective}\n")
# warp_im2_projective = warpH(img2, H2to1_projective, outsize, LAB_space=False, kind='linear')
# panoImg_projective = imageStitching(img1, warp_im2_projective)
# cv2.imwrite('bay/bay_projective.png', panoImg_projective)
#
# """BAD RESULTS"""
# img1 = cv2.imread('sintra/sintra2_pyr.png')
# img2 = cv2.imread('sintra/sintra1_pyr.png')
# outsize = np.array([1300, 2000])
# img1 = baseImageTranslation(img1, outsize,
#                                 shiftX=0,
#                                 shiftY=outsize[0] // 10)
# p1, p2 = getPoints_SIFT(img1,img2, N=30, treshhold=0.15)
# print("points")
#
# H2to1_affine = computeAffineH(p1, p2)
# print(f"H2to1_sift:\n {H2to1_affine}\n")
# warp_im2_affine = warpH(img2, H2to1_affine, outsize, LAB_space=False, kind='linear')
# panoImg_affine = imageStitching(img1, warp_im2_affine)
# cv2.imwrite('sintra/sintra_affine.png', panoImg_affine)
#
# H2to1_projective = computeH(p1, p2)
# print(f"H2to1_sift:\n {H2to1_projective}\n")
# warp_im2_projective = warpH(img2, H2to1_projective, outsize, LAB_space=False, kind='linear')
# panoImg_projective = imageStitching(img1, warp_im2_projective)
# cv2.imwrite('sintra/sintra_projective.png', panoImg_projective)