import cv2
from scipy import ndimage

from disable_enable_print import *

def util_rotate_image(image, angle):
    return ndimage.rotate(image, angle, reshape=False)

toShowOutput = True
def util_show_image(_title, img, _waittime=0, _writeToFile=1):
    
    # toShowOutput = 1
    if toShowOutput:
        logging.debug ('util_show_image called')
        cv2.imshow(_title, img)
        if not _waittime:
            cv2.waitKey(0)
        else:
            cv2.waitKey(_waittime)
        if _writeToFile:
            util_write_image(_title + '.png', img)

def util_invert_image(img):
    return cv2.bitwise_not(img)


def util_write_image(_title, img):
    cv2.imwrite(_title, img)

def util_crop_image(img, beginX, lenX, beginY, lenY):
    cropped = img[beginY:beginY + lenY, beginX:beginX + lenX]
    return cropped

def util_crop_image(img, beginX, lenX, beginY, lenY):
    cropped = img[beginY:beginY + lenY, beginX:beginX + lenX]
    return cropped

def util_read_as_gray_image(imgPath, lmt_low=70, lmt_high=255):
    img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img = cv2.inRange(img, lmt_low, lmt_high)
    return img
