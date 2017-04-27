import numpy as np
import cv2
from cv2 import cv
from imutils import perspective
import imutils

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



im = cv2.imread('/home/bhanu/projects/opencv/with_camera/two_contour_detection/bisc2.jpg')
im_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

ball_ycrcb_mint = np.array([0, 90, 100], np.uint8)
ball_ycrcb_maxt = np.array([25, 255, 255], np.uint8)
ball_ycrcb = cv2.inRange(im_ycrcb, ball_ycrcb_mint, ball_ycrcb_maxt)
# cv2.imwrite('Photos/output2.jpg', ball_ycrcb) # Second image
areaArray = []
count = 1

contours, _ = cv2.findContours(
    ball_ycrcb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    areaArray.append(area)

# cv2.fillPoly(overlay, contours, [104, 255, 255])

# first sort the array by area
sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

# find the nth largest contour [n-1][1], in this case 2
secondlargestcontour = sorteddata[1][1]

# draw it
x, y, w, h = cv2.boundingRect(secondlargestcontour)
cv2.drawContours(im, secondlargestcontour, -1, (255, 0, 0), 2)
cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), -1)

# rectangle_contour(im, secondlargestcontour)
cv2.imwrite('/home/bhanu/projects/opencv/with_camera/two_contour_detection/output.jpg', im)
