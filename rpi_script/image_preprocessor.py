
from __future__ import division
import cv2
import numpy as np
from imutils import perspective
import imutils
import platform

from disable_enable_print import *
from configuration import *
from util_image import *

glob_lowH = -1
glob_highH = -1

glob_lowS = -1
glob_highS = -1

glob_lowV = -1
glob_highV = -1

PLATFORM_MACHINE = platform.machine()

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

def shape_padding(img, padding = 20):
    shape=img.shape
    print "shape_padding ",shape
    w=shape[1]
    h=shape[0]

    base_size=h+padding,w+padding
    #make a 3 channel image for base which is slightly larger than target img
    base=np.zeros(base_size,dtype=np.uint8)
    cv2.rectangle(base,(0,0),(w+padding,h+padding),(0,0),padding)#really thick white rectangle
    base[padding/2:h+padding/2,padding/2:w+padding/2]=img #this works

    return base

def shape_transform(img, image_type):
    # input should be 4 points within image size range
    # returns the same image with transformed / caged transformed in my language
    rows, cols, ch = img.shape

    ratio = cols / 640.0    
    # ratio = 0.5
    # print img.shape
    pts1 = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
    if image_type == 'top':
        pts1 = np.float32([[76*ratio,0*ratio],[566*ratio,0*ratio],[0*ratio,443*ratio],[637*ratio,454*ratio]])
    pts2 = np.float32([[0,0],[cols,0],[0,rows+10],[cols+10,rows+10]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    # dst = cv2.resize(dst, None, fx=1 / 2, fy=1 / 2)
    # util_show_image('result after shape_transform',dst)
    return dst

def contour_to_rectangle (img, image_type):
    gray = img #assumed black and white image as input
    # cv2.imshow("mask_clean",mask_clean)

    if image_type == 'front':
        gray = util_rotate_image(gray,-43)
    if image_type == 'top':
        gray = util_rotate_image(gray,0)

    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,3)

    ret,thresh = cv2.threshold(gray,1,255,0)
    if PLATFORM_MACHINE == 'armv7l':
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #RPI
    else:
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
    for c in contours:
        rect = cv2.boundingRect(c)
        # print rect
        if rect[2] < 10 or rect[3] < 10: continue
        # print cv2.contourArea(c)
        x,y,w,h = rect
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,255),-1)
        # cv2.putText(img,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
    # cv2.imshow("Show",img)
    return gray


def process(image):
    
    global glob_lowH
    global glob_highH
    global glob_lowS
    global glob_highS
    global glob_lowV
    global glob_highV

    image_orig = image.copy()

    image = cv2.resize(image, None, fx=1 / 4, fy=1 / 4)
    #image = cv2.resize(image, None, fx=1 / 2, fy=1 / 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Blur
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

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

    return mask_clean

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

enabled_tracker = True
def main(img_path,config_data, image_type):
    
    global glob_lowH
    global glob_highH
    global glob_lowS
    global glob_highS
    global glob_lowV
    global glob_highV

    if image_type == 'front':
        glob_lowH = config_data['img_proc']['front']['lowH']
        glob_highH = config_data['img_proc']['front']['highH']

        glob_lowS = config_data['img_proc']['front']['lowS']
        glob_highS = config_data['img_proc']['front']['highS']

        glob_lowV = config_data['img_proc']['front']['lowV']
        glob_highV = config_data['img_proc']['front']['highV']
        
    elif image_type == 'top':
        glob_lowH = config_data['img_proc']['top']['lowH']
        glob_highH = config_data['img_proc']['top']['highH']

        glob_lowS = config_data['img_proc']['top']['lowS']
        glob_highS = config_data['img_proc']['top']['highS']

        glob_lowV = config_data['img_proc']['top']['lowV']
        glob_highV = config_data['img_proc']['top']['highV']

    cv2.namedWindow("Video")
    # create trackbars for color change
    if enabled_tracker:
        cv2.createTrackbar('H1', 'Video', glob_lowH, 359, onChangeH1)
        cv2.createTrackbar('S1', 'Video', glob_lowS, 256, onChangeS1)
        cv2.createTrackbar('V1', 'Video', glob_lowV, 256, onChangeV1)

        cv2.createTrackbar('H2', 'Video', glob_highH, 359, onChangeH2)
        cv2.createTrackbar('S2', 'Video', glob_highS, 256, onChangeS2)
        cv2.createTrackbar('V2', 'Video', glob_highV, 256, onChangeV2)
    

    firstCapture = True
    img = cv2.imread(img_path)
    img = shape_transform(img, image_type)
    while True:
        # f, img = video.read()
        f = True 

        """
        if firstCapture:
            firstCapture = False
            cv2.imwrite('bisc.jpg',img)
        """
        result = process(img)
        cv2.imshow('Video', result)
        util_write_image('debug_img_preproc_main_covershape.png',result)
        result = contour_to_rectangle(result,image_type) 
        result = shape_padding(result)
        util_write_image('debug_img_preproc_main_rectangles.png',result)
        # result = util_rotate_image(result,-45)
        # cv2.imshow('Video', result)

        # Wait for 1ms
        key = cv2.waitKey(1) & 0xFF
        # Press escape to exit
        if key == 27:
            return

        # Reached end of video
        if not f:
            return

def run(img_path,config_data,image_type):
    global glob_lowH
    global glob_highH
    global glob_lowS
    global glob_highS
    global glob_lowV
    global glob_highV

    # logging.debug('config_data: %s'%(config_data['img_proc']))
	
    result = ""
    img = cv2.imread(img_path)

    img = shape_transform(img, image_type)
    
    util_write_image('debug_after_shape_transform_' + image_type + '.png',img)

    if image_type == 'front':
        glob_lowH = config_data['img_proc']['front']['lowH']
        glob_highH = config_data['img_proc']['front']['highH']

        # logging.debug('glob_lowH: %s, glob_highH: %s',glob_lowH,glob_highH)

        glob_lowS = config_data['img_proc']['front']['lowS']
        glob_highS = config_data['img_proc']['front']['highS']

        glob_lowV = config_data['img_proc']['front']['lowV']
        glob_highV = config_data['img_proc']['front']['highV']
        result = process(img)
        util_write_image('debug_img_preproc_front_covershape.png',result)
        result = contour_to_rectangle(result,image_type) 
        result = shape_padding(result)
        util_write_image('debug_img_preproc_front_rectangles.png',result)
        # util_show_image('output:result',result)

    elif image_type == 'top':
        glob_lowH = config_data['img_proc']['top']['lowH']
        glob_highH = config_data['img_proc']['top']['highH']

        glob_lowS = config_data['img_proc']['top']['lowS']
        glob_highS = config_data['img_proc']['top']['highS']

        glob_lowV = config_data['img_proc']['top']['lowV']
        glob_highV = config_data['img_proc']['top']['highV']
        imgH, imgW, imgD = img.shape
        img = util_crop_image(img,0,int(imgW) - 80, 20, int(imgH) - 20)
        result = process(img)
        util_write_image('debug_img_preproc_top_covershape.png',result)
        result = contour_to_rectangle(result,image_type) 
        result = shape_padding(result)
        util_write_image('debug_img_preproc_top_rectangles.png',result)
        # util_show_image('output:result',result)
        # result = cv2.flip(result,1)
    return result

if __name__ == '__main__':
    config_data = process_config()
    config_data = config_data['dreamlite']
    # config_data = config_data['hitnrun']
    # config_data = config_data['butterbake']
    # print config_data

    main('img_local/2.jpg',config_data, 'front')
    # main('img_local/0.jpg',config_data, 'top')
    # run('img_local/2.jpg',config_data, 'front')
    # run('bugfix_images/0.10.jpg',config_data, 'top')
    # run('bugfix_images/0.20.jpg',config_data, 'top')
    # run('img_local/0.jpg',config_data, 'top')
