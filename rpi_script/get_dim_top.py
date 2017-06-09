import cv2
import image_preprocessor

from disable_enable_print import *
from util_image import *

import get_dim_l

pre_rotate_angle = 0 #TODO, change as per camera angle
def get(img_path, config_data):
    logging.debug ('Inside...%s',__name__)
    logging.debug ("TopImagePath: %s",img_path)
    ret_img = image_preprocessor.run(img_path,config_data,'top')
    # util_show_image('Preprocessed_Top_Image',ret_img)   #DEBUG_TRY

    l1, l2 = get_dim_l.get(ret_img)

    greater = 0
    if l1 > l2:
        greater = l1
    else:
        greater = l2
        
    len_mult_factor = config_data['img_proc']['top']['len_mult_factor']
    final_length = len_mult_factor*greater

    logging.debug ("Length: %s", final_length)
    return final_length

if __name__ == '__main__':
    img_path = 'debug_get_dim_l_input.jpg' #front
    img = cv2.imread(img_path) 
    h,w = get(img)
    print "H: %s, W: %s"%(h,w)