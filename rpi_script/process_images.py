import detect_dimensions
import detect_color

def process(img_dict, clr_profile):
    print "Data: ",img_dict
    print "Profile: ",clr_profile

    dim_ret = detect_dimensions.get_whl(img_dict)
    print dim_ret

    clr_ret = detect_color.get_color(clr_profile)
    print clr_ret

    return {'dimensions':dim_ret,'color':clr_ret}

    
