import cv2
import image_preprocessor

from disable_enable_print import *

import get_dim_l

pre_rotate_angle = 0 #TODO, change as per camera angle
def get(img_path):
    logging.debug ('Inside...%s',__name__)
    logging.debug ("FrontImagePath: %s",img_path)
    ret_img = image_preprocessor.run(img_path,'top')
    # cv2.imshow('Preprocessed:Top', ret_img)
    # key = cv2.waitKey(0)

    l1, l2 = get_dim_l.get(ret_img)

    greater = 0
    if l1 > l2:
        greater = l1
    else:
        greater = l2

    logging.debug ("Length: %s", greater)
    return greater