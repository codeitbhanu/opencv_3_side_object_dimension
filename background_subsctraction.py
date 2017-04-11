import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt
from measure_size_3images import FindDimensions
from time import sleep

camera_port = 0
ramp_frames = 30
camera = cv2.VideoCapture(camera_port)

def set_res(cap, x, y):
    cap.set(3, 1280)
    cap.set(4, 1024)


def takePicture(outPath):

    set_res(camera, 1280, 720)

    # Captures a single image from the camera and returns it in PIL format
    def get_image():
        retval, im = camera.read()
        return im

    # Ramp the camera - these frames will be discarded and are only used to allow v4l2
    # to adjust light levels, if necessary
    for i in xrange(ramp_frames):
        temp = get_image()
    print("Taking image...")
    # Take the actual image we want to keep
    camera_capture = get_image()
    file = outPath
    # A nice feature of the imwrite method is that it will automatically choose the
    # correct format based on the file extension you provide. Convenient!
    cv2.imwrite(file, camera_capture)

    # You'll want to release the camera, otherwise you won't be able to create a new
    # capture object until your script exits
    # del(camera)


def operateImage(imgPath):
    img = cv2.imread(imgPath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    height, width = img.shape
    print "Height {} Width {} ".format(height,width)
    return
    # cv2.imshow('output', img)
    # cv2.waitKey(0)
    img = cv2.inRange(img, 90, 255)

    cv2.imshow('output', img)
    cv2.waitKey(0)
    return

    # global thresholding
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    
    # plt.imshow(images[2 * 3 + 2], 'gray',interpolation='none')
    # plt.axis('off')
    # plt.show()

if __name__ == '__main__':

    # TODO:
    # For Loop 3 Camera Run
    takePicture('image_table.png')
    operateImage('image_table.png')

    print "op 1 done"
    # sleep(5)

    # takePicture('image_object.png')
    # operateImage('image_object.png')
    # print "op 2 done"
    # del(camera)