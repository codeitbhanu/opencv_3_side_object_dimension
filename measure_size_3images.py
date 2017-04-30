# Task
"""
1. Take Picture
2. Crop ROI
3. Invert it
4. Take Blue Threshold to strip off object, Hint: use HSV
5. Find Size 
6. -- Take Configured Samples from Two Side Images
7. -- List Down All Values
"""


# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

toShowOutput = False


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def util_show_image(_title, img, _waittime=0, _writeToFile=1):

    # toShowOutput = 1
    if toShowOutput:
        print 'util_show_image called'
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

class FindDimensions():
    def __init__(self, image1, image2, imageTop):
        self.imageFront = image1
        self.imageRear = image2
        self.imageTop = imageTop

        # print "imageFront: ",self.imageFront
        # print "imageRear: ",self.imageRear
        # print "imageTop: ",self.imageTop

        hf, wf = self.findRectHW(self.imageFront)
        print "imageFront - Height: {:.1f}mm Width: {:.1f}mm".format(hf, wf)

        hr, wr = self.findRectHW(self.imageRear)
        print "imageRear - Height: {:.1f}mm Width: {:.1f}mm".format(hr, wr)

        l1, l2 = self.findLength(self.imageTop)

        greater = 0
        if l1 > l2:
            greater = l1
        else:
            greater = l2

        print "imageTop - Length: {:.1f}mm".format(greater)

    def findRectHW(self, img):
        # load the image, convert it to grayscale, and blur it slightly
        image = cv2.imread(img)

        util_show_image('original', image)

        image = cv2.bitwise_not(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=1)
        edged = cv2.erode(edged, None, iterations=1)

        # find contours in the edge map
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # sort the contours from left-to-right and initialize the
        # 'pixels per metric' calibration variable
        (cnts, _) = contours.sort_contours(cnts)
        pixelsPerMetric = None

        # loop over the contours individually
        _count = 0
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if cv2.contourArea(c) < 100:
                continue

            # compute the rotated bounding box of the contour
            orig = image.copy()
            box = cv2.minAreaRect(c)
            box = cv2.cv.BoxPoints(
                box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # order the points in the contour such that they appear
            # in top-left, top-right, bottom-right, and bottom-left
            # order, then draw the outline of the rotated bounding
            # box
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
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
                pixelsPerMetric = dB / args["width"]

            # compute the size of the object
            dimA = dA / pixelsPerMetric
            dimB = dB / pixelsPerMetric
            # if _count > 0:
            # print "Object {} Width is {:.1f}mm, Height is
            # {:.1f}mm".format(_count, dimA, dimB)

            _count = _count + 1

            # draw the object sizes on the image
            cv2.putText(orig, "{:.1f}in".format(dimA),
                        (int(tltrX - 15), int(tltrY - 10)
                         ), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)
            cv2.putText(orig, "{:.1f}in".format(dimB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2)

            # show the output image
            util_show_image("Image", orig)

        return dimA, dimB

    def findLength(self, img):
        # load the image, convert it to grayscale, and blur it slightly
        image = cv2.imread(img)
        image = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.Canny(image, 1, 100)

        image = cv2.dilate(image, None, iterations=2)
        image = cv2.erode(image, None, iterations=2)

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
                cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

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
                pixelsPerMetric = dB / args["width"]

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
            util_show_image("Image", orig)

        return dimA, dimB

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #  find colors  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class FindColors():
    def __init__(self, image1, image2):  # only back and front size color
        self.imageFront = image1
        self.imageRear = image2
        self.defaultConfig = 0

        self.findFrontColor(self.imageFront, self.defaultConfig)
        # IMPORTANT Book Text Detector:
        # image = cv2.imread(self.imageFront)
        # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ## retVal, threshold = cv2.threshold(gray,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        # util_show_image('adaptiveThreshold',threshold)

    def findFrontColor(self, img, config=0):
        # load the image
        image = cv2.imread(img)
        res = cv2.GaussianBlur(image, (15, 15), 0)
        # util_show_image('Gaussion Blur',res)
        res = cv2.medianBlur(res, 15)
        res = cv2.dilate(res, None, iterations=5)
        image = res

        # define the list of boundaries
        boundaries = [
            #B, G, R
            ([56, 66, 94], [171, 201, 218])
            # ([86, 31, 4], [220, 88, 50]),
            # ([25, 146, 190], [62, 174, 250]),
            # ([103, 86, 65], [145, 133, 128])
        ]

        # loop over the boundaries
        for (lower, upper) in boundaries:
            # create NumPy arrays from the boundaries
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv2.inRange(image, lower, upper)
            output = cv2.bitwise_and(image, image, mask=mask)

            # show the images
            cv2.imshow("images", output)
            cv2.waitKey(0)
            return [55, 66, 77]  # BGR values

    def findRearColor(self, img, config=0):
        return [99, 89, 79]  # BGR values


# construct the argument parse and parse the arguments


if __name__ == '__main__':

    # TODO:
    # For Loop 3 Camera Run

    ap = argparse.ArgumentParser()
    ap.add_argument("-if", "--imageFront", required=False,
                    help="path to the front input image")
    ap.add_argument("-ir", "--imageRear", required=False,
                    help="path to the rear input image")
    ap.add_argument("-it", "--imageTop", required=False,
                    help="path to the top input image")
    ap.add_argument("-o", "--showOutput", required=False,
                    help="choice yes/no if output to watch")
    ap.add_argument("-w", "--width", type=float, required=True,
                    help="width of the left-most object in the image (in inches)")
    args = vars(ap.parse_args())
    # print args

    # TODO: Validation Check
    toShowOutput = True
    if args['showOutput'] == 'yes':
        toShowOutput = True
    dim = FindDimensions(args['imageFront'], args['imageRear'], args['imageTop'])
    # clr = FindColors(args['imageFront'],args['imageRear'])
    
    #init
    
    # img = cv2.imread(args['imageFront'])
    # inv_img = util_invert_image(img)
    # util_show_image('inverted',inv_img)
    
