import cv2
import numpy as np

im = cv2.imread("wood.jpg")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# inverted threshold (light obj on dark bg)
_, bin = cv2.threshold(gray, 120, 255, 1)
bin = cv2.dilate(bin, None)  # fill some holes
bin = cv2.dilate(bin, None)
bin = cv2.erode(bin, None)   # dilate made our shape larger, revert that
bin = cv2.erode(bin, None)
contours = cv2.findContours(bin, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

rc = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rc)
for p in box:
    pt = (p[0], p[1])
    print pt
    cv2.circle(im, pt, 5, (200, 0, 0), 2)
cv2.imshow("wood", im)
cv2.waitKey()
