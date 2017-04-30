from __future__ import division
import cv2
import numpy as np
from imutils import perspective
import imutils

# define the list of boundaries
lower = [44, 51, 60]
upper = [214, 255, 248]

def rotateImage(image, angle):
    from scipy import ndimage
    return ndimage.rotate(image, angle, reshape=False)


# load the image
image = cv2.imread('if.jpg')
# image = cv2.imread('p.png')
image = rotateImage(image, -45)

#CAUTION: THIS SHOULD BE ADJUSTED ON REAL SETUP
top_left_x = 500
top_left_y = 30
bottom_right_x = 1150
bottom_right_y = 700

crop_img = image[30:700, 500:1150] # Crop from x, y, w, h -> 100, 200, 300, 400
# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

crop_img[np.where((crop_img==[0,0,0]).all(axis=2))] = [255,150,70]

# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)


# cv2.imwrite('roi_if.jpg',base_image)


# loop over the boundaries
counter = 0
# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(crop_img, lower, upper)
output = cv2.bitwise_and(crop_img, crop_img, mask=mask)

# show the images
# cv2.imshow("images", np.hstack([image, output]))
cv2.imshow("images", output)
# cv2.imwrite('img_'+str(counter)+'.png',output)
counter = counter + 1
cv2.waitKey(0)
