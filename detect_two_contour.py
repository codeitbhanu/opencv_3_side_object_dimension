import numpy as np
import cv2
from cv2 import cv
from imutils import perspective
import imutils


def rotateImage(image, angle):
    image
    image_center = tuple(np.array(image.shape) / 2)
    print'image_center: ',image_center
    image_center = (image_center[0],image_center[1])
    print'now image_center: ',image_center
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1)
    result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
    return result

def rectangle_contour(image, contour, toFill):
    
    if contour is None:
        return image

    if toFill is True:
        toFill = -1
    else:
        toFill = 2

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
    cv2.drawContours(image_with_rect, [box.astype("int")], -1, (0, 0, 255), toFill)
    return image_with_rect


'''/home/bhanu/projects/opencv/with_camera/two_contour_detection/'''
im = cv2.imread('bisc2.jpg')
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
print "Number of Found Contours: ",len(sorteddata)    # 3 rows in your example
# find the nth largest contour [n-1][1], in this case 2
secondLargestContour = sorteddata[1][1]
print(np.matrix(secondLargestContour))


# draw it
"""
x, y, w, h = cv2.boundingRect(secondLargestContour)
cv2.drawContours(im, secondLargestContour, -1, (0, 0, 255), 2)
cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), -1)
"""
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = rectangle_contour(im, secondLargestContour, True)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# Now create a mask of logo and create its inverse mask also
img_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
cv2.imshow('output.jpg',img_gray)
cv2.waitKey(0)
"""
img_rgb(img_gray.size(), CV_8UC3);


  // convert grayscale to color image
  cv::cvtColor(img_gray, img_rgb, CV_GRAY2RGB);
"""
im = imutils.rotate(img_gray, -45)
cv2.imwrite('output.jpg', im)
