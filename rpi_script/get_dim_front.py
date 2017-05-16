import cv2
from disable_enable_print import *
from util_image import *
import image_preprocessor

import get_dim_hw

pre_rotate_angle = 0 #TODO, change as per camera angle
def get(img_path,config_data):
    logging.debug ('Inside...%s',__name__)
    logging.debug ("FrontImagePath: %s",img_path)
    ret_img = image_preprocessor.run(img_path,config_data,'front')
    # util_show_image('Preprocessed_Front_Image',ret_img)

    h,w = get_dim_hw.get(ret_img)

    logging.debug ("Front View Height: %s Width: %s",h,w)
    return w,h