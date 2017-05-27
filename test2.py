import cv2
import numpy as np


stitched = cv2.imread('bugfix_images/test1.png', 0)
(_, mask) = cv2.threshold(stitched, 1.0, 255.0, cv2.THRESH_BINARY);

# findContours destroys input
temp = mask.copy()
(contours, _) = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours by largest first (if there are more than one)
contours = sorted(contours, key=lambda contour:len(contour), reverse=True)
roi = cv2.boundingRect(contours[0])

print roi

# use the roi to select into the original 'stitched' image
out = stitched[roi[1]:roi[3], roi[0]:roi[2]]

cv2.imshow('output',out)
cv2.waitKey(0)