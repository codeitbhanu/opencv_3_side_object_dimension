
from __future__ import division
import cv2
import numpy as np
from imutils import perspective
import imutils

glob_h1 = 10
glob_h2 = 230

glob_s1 = 10
glob_s2 = 150

glob_v1 = 44
glob_v2 = 250

low_brown = (10,10,44)
brown = (230,150,250)

def process(image):
    
    global glob_h1
    global glob_h2
    global glob_s1
    global glob_s2
    global glob_v1
    global glob_v2

    image = cv2.resize(image, None, fx=1 / 2, fy=1 / 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Blur
    # image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Filter by colour
    # 0-10 hue                 H,  S,   V
    mask = cv2.inRange(image_blur_hsv, low_brown, brown)
    image = cv2.bitwise_and(image_blur_hsv,image_blur_hsv,mask=mask)
    # cv2.drawContours(im, secondLargestContour, -1, 255, -1)

    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def onChangeH1(x):
    global glob_h1
    glob_h1 = x


def onChangeS1(x):
    global glob_s1
    glob_s1 = x


def onChangeV1(x):
    global glob_v1
    glob_v1 = x


def onChangeH2(x):
    global glob_h2
    glob_h2 = x


def onChangeS2(x):
    global glob_s2
    glob_s2 = x


def onChangeV2(x):
    global glob_v2
    glob_v2 = x




# im = cv2.imread('if.jpg')    
# im = process(im)
# cv2.imshow('Result',im)
# cv2.waitKey(0)



enabled_tracker = False
def main():
    """
    # Load video
    video = cv2.VideoCapture(2)

    if not video.isOpened():
        video.release()
        raise RuntimeError('Video not open')
    """
    cv2.namedWindow("Video")
    # create trackbars for color change
    if enabled_tracker:
        cv2.createTrackbar('H1', 'Video', glob_h1, 359, onChangeH1)
        cv2.createTrackbar('S1', 'Video', glob_s1, 256, onChangeS1)
        cv2.createTrackbar('V1', 'Video', glob_v1, 256, onChangeV1)

        cv2.createTrackbar('H2', 'Video', glob_h2, 359, onChangeH2)
        cv2.createTrackbar('S2', 'Video', glob_s2, 256, onChangeS2)
        cv2.createTrackbar('V2', 'Video', glob_v2, 256, onChangeV2)
    

    firstCapture = True
    while True:
        # f, img = video.read()
        f = True
        # img = cv2.imread('bisc.jpg')    
        img = cv2.imread('if.jpg')    

        """
        if firstCapture:
            firstCapture = False
            cv2.imwrite('bisc.jpg',img)
        """
        result = process(img)

        # result = rotateImage(result,-45)
        cv2.imshow('Video', result)

        # Wait for 1ms
        key = cv2.waitKey(1) & 0xFF

        # Press escape to exit
        if key == 27:
            return

        # Reached end of video
        if not f:
            return


if __name__ == '__main__':
    main()