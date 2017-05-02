import get_dim_front as gdf
import get_dim_top as gdt
from disable_enable_print import *

def get_whl(img_dict):
    logging.debug("Processing For Dimensions...")
    # camera_port_front = 0 #img_front
    # camera_port_rear = 1 #img_rear
    # camera_port_top = 2 #img_top
    w,h = gdf.get(img_dict[0])
    l = gdt.get(img_dict[2])

    ret = {  'w':w,
            'h':h,
            'l':l}
    
    return ret