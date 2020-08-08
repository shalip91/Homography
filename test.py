import cv2
import numpy as np
import glob


RC = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
])

K=np.array([
    [500, 0, 400],
    [0, 500, 400],
    [0, 0, 1]
])

print(K @ RC)