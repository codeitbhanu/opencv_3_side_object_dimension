import detect_dimensions
import detect_color

from disable_enable_print import *

def process(img_dict, clr_profile):
    print "Data: ",img_dict
    print "Profile: ",clr_profile

    # disable_print()

    dim_ret = detect_dimensions.get_whl(img_dict)
    print dim_ret

    
    # enable_print()

    clr_ret = detect_color.get_color(img_dict, profile=clr_profile)
    print clr_ret

    return {'dimensions':dim_ret,'color':clr_ret}

    
