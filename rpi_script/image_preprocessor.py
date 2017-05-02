
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

def rotateImage(image, angle):
    image
    image_center = tuple(np.array(image.shape) / 2)
    # print'image_center: ', image_center
    image_center = (image_center[0], image_center[1])
    # print'now image_center: ', image_center
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1)
    result = cv2.warpAffine(image, rot_mat, image.shape,
                            flags=cv2.INTER_LINEAR)
    return result


def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):

    # Copy
    image = image.copy()
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour)
                     for contour in contours]

    if not contour_sizes:
        return None, image

    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    # cv2.drawContours(mask, [biggest_contour], -1, 255, -1) #wanted rotated
    # box around biscuit
    cv2.drawContours(mask, [biggest_contour], 0, (0, 0, 255), 2)
    return biggest_contour, mask

def rectangle_contour(image, contour):
    
    if contour is None:
        return image

    # Bounding rectangle
    image_with_rect = image.copy()
    """ # OLD LOGIC RECT NON-ROTATING
    # rect = cv2.boundingRect(contour)

    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_with_rect, (x, y), (x + w, y + h), green, 2, cv2.CV_AA)
    return image_with_rect
    """
    # compute the rotated bounding box of the contour
        # image_with_rect = image.copy()
    box = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(image_with_rect, [box.astype("int")], -1, (0, 255, 0), 1)
    return image_with_rect


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
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # Filter by colour
    # 0-10 hue                 H,  S,   V
    min_bisc_brown = np.array([glob_h1, glob_s1, glob_v1])
    max_bisc_brown = np.array([glob_h2, glob_s2, glob_v2])
    mask1 = cv2.inRange(image_blur_hsv, min_bisc_brown, max_bisc_brown)

    # 170-180 hue
    min_bisc_brown2 = np.array([glob_h1 + 170, glob_s1, glob_v1])
    max_bisc_brown2 = np.array([glob_h2 + 180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_bisc_brown2, max_bisc_brown2)

    # Combine masks
    mask = mask1 + mask2

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    return mask_clean

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


# if __name__ == '__main__':
#     main()


def run(img_path,image_type):
    img = cv2.imread(img_path)    
    result = process(img)
    if image_type == 'front':
        result = rotateImage(result,-45)
    return result