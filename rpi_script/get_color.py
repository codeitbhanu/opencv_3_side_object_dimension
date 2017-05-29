
from __future__ import division
import cv2
import numpy as np
from imutils import perspective
import imutils
import platform

from PIL import Image
from scipy.ndimage import median_filter
from time import sleep


from disable_enable_print import *
from configuration import *
from util_image import *


PLATFORM_MACHINE = platform.machine()

blue = (255, 0, 0)
green = (0, 255, 0)
low_green = (0, 254, 0)
red = (0, 0, 255)
low_red = (0, 0, 254)
# high_red = (0, 0, 255)
black = (0, 0, 0)
white = (255,255,255)

glob_lowH = -1
glob_highH = -1

glob_lowS = -1
glob_highS = -1

glob_lowV = -1
glob_highV = -1

imageType=0

def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    
    # Copy
    image = image.copy()
    if PLATFORM_MACHINE == 'armv7l':
        _,contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #RPI
    else:
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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
    # util_show_image('biggest_contour',biggest_contour)
    return biggest_contour, mask
    
def rectangle_contour(image, contour, toFill=True, colorFill=green):
    
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
    cv2.drawContours(image_with_rect, [box.astype("int")], -1, (0, 255, 0), 1)
    # util_show_image('image_with_rect',image_with_rect)
    return image_with_rect, box


def process(image):
    
    global glob_lowH
    global glob_highH
    global glob_lowS
    global glob_highS
    global glob_lowV
    global glob_highV

    image = cv2.resize(image, None, fx=1 / 2, fy=1 / 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Blur
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    """
    # 0-10 hue
    min_red = np.array([0, 100, 80])
    max_red = np.array([20, 256, 256])
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

    # 170-180 hue
    min_red2 = np.array([170, 100, 80])
    max_red2 = np.array([190, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

    """
    print "H1: %s S1: %s V1: %s -------- H2: %s S2: %s V2: %s"%(glob_lowH,glob_lowS,glob_lowV,glob_highH,glob_highS,glob_highV)
    # Filter by colour
    # 0-10 hue                 H,  S,   V
    min_bisc_brown = np.array([glob_lowH, glob_lowS, glob_lowV])
    max_bisc_brown = np.array([glob_highH, glob_highS, glob_highV])
    mask1 = cv2.inRange(image_blur_hsv, min_bisc_brown, max_bisc_brown)

    # 170-180 hue
    min_bisc_brown2 = np.array([glob_lowH + 170, glob_lowS, glob_lowV])
    max_bisc_brown2 = np.array([glob_highH + 180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_bisc_brown2, max_bisc_brown2)

    # Combine masks
    mask = mask1 + mask2

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Find biggest biscuit
    big_biscuit_contour, mask_biscuit = find_biggest_contour(
        mask_clean)

    # Overlay cleaned mask on image
    overlay = overlay_mask(mask_clean, image)

    # Circle biggest biscuit
    # circled = circle_contour(overlay, big_biscuit_contour)
    rectangled, box = rectangle_contour(overlay, big_biscuit_contour)

    # logging.debug ("rectangled %s",rectangled)
    # Finally convert back to BGR to display
    bgr = cv2.cvtColor(rectangled, cv2.COLOR_RGB2BGR)
    # bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)

    # return bgr

    # util_show_image('bgr',bgr)
    return mask_closed,mask_clean,box

def onChangeH1(x):
    global glob_lowH
    glob_lowH = x


def onChangeS1(x):
    global glob_lowS
    glob_lowS = x


def onChangeV1(x):
    global glob_lowV
    glob_lowV = x


def onChangeH2(x):
    global glob_highH
    glob_highH = x


def onChangeS2(x):
    global glob_highS
    glob_highS = x


def onChangeV2(x):
    global glob_highV
    glob_highV = x

def fill_black_with_white(data):
    data[np.where((data == [0,0,0]).all(axis = 2))] = [255,255,255]
    return data

def get_color_code(image):
    # util_show_image('input image for get color',image)
    global imageType
    crop_percent = 10
    h,w,d = image.shape
    xstart,xend = (w*(crop_percent)/100),(w-(w*(crop_percent)/100))
    ystart,yend = (h*(crop_percent)/100),(h-(h*(crop_percent)/100))

    xstart = int(round(xstart))
    xend = int(round(xend))
    ystart = int(round(ystart))
    yend = int(round(yend))

    image = image[ystart:yend,xstart:xend]

    # util_write_image('debug_get_color_%s.jpg'%format(imageType),image)
    # util_show_image('After Cropping...',image)

    average_color_per_row = np.average(image, axis=0)
    average_color = np.mean(average_color_per_row, axis=0)
    logging.debug ("get_color_code: Average Color: %s",average_color)
    average_color = np.uint8(average_color)
    average_color_img = np.array([[average_color]*100]*100, np.uint8)
    # logging.debug(average_color_img)
    # cv2.imwrite( "average_color.png", average_color_img )
    
    # util_show_image('average_color.png',average_color_img)  #DEBUG_TRY
    return average_color


enabled_tracker = False #DEBUG_TRY
def find_color(img_path, imgType='unknown', clr_profile=0):
    # create trackbars for color change
    global glob_lowH
    global glob_highH
    global glob_lowS
    global glob_highS
    global glob_lowV
    global glob_highV

    if enabled_tracker:
        cv2.namedWindow("Video")
        cv2.createTrackbar('H1', 'Video', glob_lowH, 359, onChangeH1)
        cv2.createTrackbar('S1', 'Video', glob_lowS, 256, onChangeS1)
        cv2.createTrackbar('V1', 'Video', glob_lowV, 256, onChangeV1)

        cv2.createTrackbar('H2', 'Video', glob_highH, 359, onChangeH2)
        cv2.createTrackbar('S2', 'Video', glob_highS, 256, onChangeS2)
        cv2.createTrackbar('V2', 'Video', glob_highV, 256, onChangeV2)

    firstCapture = True
    while firstCapture:
        firstCapture = False    #DEBUG_TRY
        # sleep(0.100)
        # f, img = video.read()
        f = True
        # img = cv2.imread('bisc.jpg')    
        img = cv2.imread(img_path)    
        img_orig = img.copy()

        _,mask_clean,box = process(img)

        mask_clean = util_rotate_image(mask_clean,-45)
        # mask_clean = cv2.dilate(mask_clean, None, iterations=2)
        mask_clean = cv2.erode(mask_clean, None, iterations=2)

        img_resized = cv2.resize(img_orig, None, fx=1 / 2, fy=1 / 2)
        img_resized = util_rotate_image(img_resized,-45)
        # cv2.drawContours(img_resized, [box.astype("int")], -1, (0, 255, 0), 1)

        result = cv2.bitwise_and(img_resized,img_resized,mask = mask_clean)
        # result = fill_black_with_white(result)

        # cv2.imwrite('example.jpg',result)
        gray=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(result, 10, 250)
        if PLATFORM_MACHINE == 'armv7l':
            (_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #RPI
        else:
            (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        idx = 0
        new_img = None
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w>50 and h>50:
                idx+=1
                new_img=result[y:y+h,x:x+w]
                cv2.imwrite('front'+str(idx) + '.png', new_img)
        result = new_img

        # util_show_image('result--------------',result)

        height, width, depth = result.shape
        # logging.debug ("Height: %s Width: %s Depth: %s",height,width,depth)

        # TODO: To Add More Color Picking Profile Support
        # TODO: Add Logic to Pick Color Except Black
        if not imgType == 'rear':
            result = util_invert_image(result)
        debug_file_name = 'debug_get_color_'+str(imgType)+'.jpg'
        util_write_image(debug_file_name, result)
        b = g = r = 0
        if clr_profile == 0:
            b,g,r = get_color_code(result)

        global imageType
        if imageType == 'front':
            result = util_rotate_image(result,3)
        if imageType == 'rear':
            result = util_rotate_image(result,-2)
        
        #DEBUG_TRY
        """
        cv2.imshow('Video', result)

        # Wait for 1ms
        key = cv2.waitKey(0) & 0xFF

        # Press escape to exit
        if key == 27:
            return

        # Reached end of video
        if not f:
            return
        """
        return result,r,g,b
        

def get(img_path, config_data, clr_profile=0, imgType=None):
    # enable_print()
    global imageType
    imageType = imgType

    global glob_lowH
    global glob_highH
    global glob_lowS
    global glob_highS
    global glob_lowV
    global glob_highV
    
    result = ""
    if imgType == 'front':
        glob_lowH = config_data['get_color']['front']['lowH']
        glob_highH = config_data['get_color']['front']['highH']

        glob_lowS = config_data['get_color']['front']['lowS']
        glob_highS = config_data['get_color']['front']['highS']

        glob_lowV = config_data['get_color']['front']['lowV']
        glob_highV = config_data['get_color']['front']['highV']

        result,r,g,b =  find_color(img_path, imgType, clr_profile)
        # h,w,d = result.shape
        # result = util_crop_image(result,0,w,20,h-20)
        # result = util_rotate_image(result,3)
        # util_show_image('output:result',result)

    elif imgType == 'rear':
        glob_lowH = config_data['get_color']['rear']['lowH']
        glob_highH = config_data['get_color']['rear']['highH']

        glob_lowS = config_data['get_color']['rear']['lowS']
        glob_highS = config_data['get_color']['rear']['highS']

        glob_lowV = config_data['get_color']['rear']['lowV']
        glob_highV = config_data['get_color']['rear']['highV']

        # imgH, imgW, imgD = img.shape
        # img = util_crop_image(img,0,int(imgW) - 80, 20, int(imgH) - 20)
        result,r,g,b =  find_color(img_path, imgType, clr_profile)
        # util_show_image('output:result',result)
        # result = cv2.flip(result,1)


    return result,r,g,b

if __name__ == '__main__':
    config_data = process_config()
    config_data = config_data['type1']

    # get('img_local/0.jpg',config_data, 0,'front')
    get('img_local/1.jpg',config_data, 0,'rear')
