import cv2
import numpy as np
# import bgsegm

cap = cv2.VideoCapture(0)
# fgbg = cv2.createBackgroundSubtractorMOG()
fgbg = cv2.BackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow('orig',frame)
    cv2.imshow('fg',fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()