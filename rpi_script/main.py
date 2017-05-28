import werkzeug_server
import flask_server


from disable_enable_print import *
from configuration import *

import capture

import process_images

camera_port_front = 2 #img_front
camera_port_rear = 1 #img_rear
camera_port_top = 0 #img_top

clr_profile = 0

global config_data 

if __name__ == '__main__':
    # werkzeug_server.run_server('enp0s26u1u1')
    # flask_server.run_server('enp0s26u1u1')
    global config_data
    logging.debug ("...inside main loop...")
    config_data = process_config()
    config_data = config_data['type1']
    # logging.debug ("Config Data: %s"%(config_data['type1'])
    
    img_dict = capture.capture_images(camera_port_front,camera_port_rear,camera_port_top)
    logging.debug (img_dict)

    ret = process_images.process(img_dict, config_data, camera_port_front,camera_port_rear,camera_port_top,clr_profile)
    # TODO: Return Type to Decide, dimension per image, Color Per Image
    # (w,h,l,#FFEEDD)

    # enable_print()
    logging.info ("Processed Data: %s",ret)
    
    #TODO: Push Values to WebApp
    #TODO: Push Values to mySql
