# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

toShowOutput = False

from disable_enable_print import *

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def util_show_image(_title, img, _waittime=0, _writeToFile=1):

    if toShowOutput:
        logging.debug ('util_show_image called')
        cv2.imshow(_title, img)
        if not _waittime:
            cv2.waitKey(0)
        else:
            cv2.waitKey(_waittime)
        if _writeToFile:
            util_write_image(_title + '.png', img)

def util_crop_image(img, beginX, lenX, beginY, lenY):
    cropped = img[beginY:beginY + lenY, beginX:beginX + lenX]
    return cropped

def util_read_as_gray_image(imgPath, lmt_low=70, lmt_high=255):
    img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img = cv2.inRange(img, lmt_low, lmt_high)
    return img

def util_write_image(_title, img):
    cv2.imwrite(_title, img)

def util_invert_image(img):
    return cv2.bitwise_not(img)



ref_width = 40
def get(image):
    # load the image, convert it to grayscale, and blur it slightly
    # image = cv2.imread(img)
    # util_show_image("Original Image", image)
    # image = image.copy()

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image = cv2.Canny(image, 1, 100)

    # image = cv2.dilate(image, None, iterations=2)
    # image = cv2.erode(image, None, iterations=2)

    edged = image

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually
    _count = 0
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 200:
            continue

        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(
            box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 2, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None:
            pixelsPerMetric = dB / ref_width

        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        _count = _count + 1

        # draw the object sizes on the image
        cv2.putText(orig, "{:.1f}mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)
                        ), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

        # show the output image
        # util_show_image("Image", orig)

    return round(dimA,2), round(dimB,2)