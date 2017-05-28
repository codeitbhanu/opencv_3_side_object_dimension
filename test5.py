import cv2
import numpy as np

import os

# os.chdir('C:/Users/gennady.nikitin/Dropbox/Coding/OpenCV')
os.chdir('/home/bhanu/projects/opencv/pyimagesearch/downloads/pokedex-find-screen-part-2/pokedex-find-screen-part-2')

img = cv2.imread('0.jpg')

# img = cv2.resize(img, None, fx=1 / 2, fy=1 / 2)

rows, cols, ch = img.shape    
print img.shape
pts1 = np.float32([[76,0],[566,0],[0,443],[637,454]])
pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img, M, (cols, rows))
cv2.imshow('My Zen Garden', dst)
cv2.imwrite('zen.jpg', dst)
cv2.waitKey()