import werkzeug_server
import flask_server


from disable_enable_print import *
import capture

import process_images

clr_profile = 0

if __name__ == '__main__':
    # werkzeug_server.run_server('enp0s20u3')
    # flask_server.run_server('enp0s20u3')

    disable_print()

    img_dict = capture.capture_images()
    logging.debug (img_dict)

    ret = process_images.process(img_dict,clr_profile)
    # TODO: Return Type to Decide, dimension per image, Color Per Image
    # (w,h,l,#FFEEDD)

    enable_print()
    logging.debug ("Processed Data: %s",ret)

    #TODO: Push Values to WebApp
    #TODO: Push Values to mySql