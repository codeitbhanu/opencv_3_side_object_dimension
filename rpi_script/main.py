import werkzeug_server
import flask_server


from disable_enable_print import *
import capture

import process_images

camera_port_front = 2 #img_front
camera_port_rear = 0 #img_rear
camera_port_top = 1 #img_top

clr_profile = 0

if __name__ == '__main__':
    # werkzeug_server.run_server('enp0s26u1u1')
    # flask_server.run_server('enp0s26u1u1')

    print "...inside main loop..."
    # disable_print()

    img_dict = capture.capture_images(camera_port_front,camera_port_rear,camera_port_top)
    logging.debug (img_dict)

    ret = process_images.process(img_dict,camera_port_front,camera_port_rear,camera_port_top,clr_profile)
    # TODO: Return Type to Decide, dimension per image, Color Per Image
    # (w,h,l,#FFEEDD)

    # enable_print()
    logging.info ("Processed Data: %s",ret)

    #TODO: Push Values to WebApp
    #TODO: Push Values to mySql
