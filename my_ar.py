import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt
from HW4 import my_homography as mh


#Add imports if needed:

#end imports

#Add functions here:
def trapezoidToRect(p2):

    pts = np.array(tuple(zip(p2[0], p2[1])))

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    p1 = np.array([
        [0, maxWidth - 1, maxWidth - 1, 0],
        [0, 0, maxHeight - 1, maxHeight - 1]
    ], dtype="float32")

    outsize = np.array([maxHeight, maxWidth], np.uint16)

    return p1, outsize
#Functions end

# HW functions:
def create_ref(im_path):
    im2 = cv2.imread(im_path)
    plt.imshow(im2)
    p2 = np.array(plt.ginput(4, 0)).T
    plt.close()

    p1, outsize = trapezoidToRect(p2)

    H2to1 = mh.ransacH(p1, p2, 1000, 1)

    ref_image = mh.warpH(im2, H2to1, outsize, LAB_space=False, kind='linear')

    return ref_image


def im2im(im1, im2, im2Path):
    img_template = create_ref(im2Path + '.jpeg')
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    img_template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    p1, p2 = mh.getPoints(im1_gray, img_template_gray, 4)
    H2to1 = mh.ransacH(p1, p2, 1000, 1)
    outsize = np.array([im1.shape[0], im1.shape[1]])
    deVinchiWarped = mh.warpH(img_template, H2to1, outsize, LAB_space=False, kind='linear')
    books_combine = mh.imageStitching(im1, deVinchiWarped)

    return books_combine

if __name__ == '__main__':
    print('my_ar')


"""3.1"""
img = create_ref("books/god_delusion.jpeg")
cv2.imwrite("books/god_delusion_warped.png", img)
cv2.imshow('img', img)
cv2.waitKey(0)

"""3.2"""

"""desk"""
# im1 = cv2.imread('books/god_delusion.jpeg')
# im2 = cv2.imread('books/de_vinci_code.jpeg')
# books_combine = im2im(im1, im2, 'books/de_vinci_code')
# cv2.imwrite("books/books_combine_desk.png", books_combine)
#
# """lemons"""
# im1 = cv2.imread('books/god_delusion_lemons.jpeg')
# im2 = cv2.imread('books/de_vinci_code.jpeg')
# books_combine = im2im(im1, im2, 'books/de_vinci_code')
# cv2.imwrite("books/books_combine_lemons.png", books_combine)
#
# """refrigirator"""
# im1 = cv2.imread('books/god_delusion_refrigerator.jpeg')
# im2 = cv2.imread('books/de_vinci_code.jpeg')
# books_combine = im2im(im1, im2, 'books/de_vinci_code')
# cv2.imwrite("books/books_combine_refrigerator.png", books_combine)


"""3.4"""