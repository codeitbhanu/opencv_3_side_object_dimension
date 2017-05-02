import cv2
from disable_enable_print import *
import image_preprocessor

import get_dim_hw

pre_rotate_angle = 0 #TODO, change as per camera angle
def get(img_path):
    logging.debug ('Inside...%s',__name__)
    logging.debug ("FrontImagePath: %s",img_path)
    ret_img = image_preprocessor.run(img_path,'front')
    # cv2.imshow('Preprocessed:Top', ret_img)
    # key = cv2.waitKey(0)

    h,w = get_dim_hw.get(ret_img)

    logging.debug ("Height: %s Width: %s",h,w)
    return w,h