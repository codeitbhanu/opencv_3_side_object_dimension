import detect_dimensions
import detect_color

from disable_enable_print import *

def process(img_dict, fport, rport, tport, clr_profile):
    logging.debug('process called')
    logging.debug("Data: %s",img_dict)
    logging.debug("Profile: %s",clr_profile)

    # disable_logging.debug()

    dim_ret = detect_dimensions.get_whl(img_dict, fport, rport, tport)
    logging.debug (dim_ret)

    
    # enable_logging.debug()

    clr_ret = detect_color.get_color(img_dict, fport, rport, tport, profile=clr_profile)
    logging.debug (clr_ret)

    return {'dimensions':dim_ret,'color':clr_ret}

    
