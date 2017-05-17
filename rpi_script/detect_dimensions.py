import get_dim_front as gdf
import get_dim_top as gdt
from disable_enable_print import *

def get_whl(img_dict,config_data,fport,rport,tport):
    logging.debug("Processing For Dimensions...")
    # camera_port_front = 0 #img_front
    # camera_port_rear = 1 #img_rear
    # camera_port_top = 2 #img_top
    w,h = gdf.get(img_dict[fport],config_data)
    l = gdt.get(img_dict[tport],config_data)

    ret = {  'w':str(w),
            'h':str(h),
            'l':str(l)}
    
    return ret
