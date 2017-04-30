import cv2
import image_preprocessor

import get_dim_hw

pre_rotate_angle = 0 #TODO, change as per camera angle
def get(img_path):
    print 'Inside...',__name__
    print "FrontImagePath: ",img_path
    ret_img = image_preprocessor.run(img_path,'front')
    # cv2.imshow('Preprocessed:Top', ret_img)
    # key = cv2.waitKey(0)

    h,w = get_dim_hw.get(ret_img)

    print "Height: ",h," Width: ",w
    return w,h