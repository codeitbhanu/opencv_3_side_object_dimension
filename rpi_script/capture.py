import cv2
import time
import platform

from disable_enable_print import *

capture_from_camera = False

laptop_capture = False

PLATFORM_MACHINE = platform.machine()

led_enabled = False
if PLATFORM_MACHINE == 'armv7l':
    led_enabled = True
else:
    led_enabled = False

#camera_port_front = 2 #img_front
#camera_port_rear = 1 #img_rear
#camera_port_top = 0 #img_top
if led_enabled:
    import RPi.GPIO as GPIO

images_local_folder_path = "img_local/"
images_cam_folder_path = "img_webcam/"

RelayPin = 11

def act_led():
    GPIO.setmode(GPIO.BOARD) # Set GPIO as numbering
    GPIO.setup(RelayPin, GPIO.OUT)
    GPIO.output(RelayPin, GPIO.LOW)

def deact_led():
    GPIO.output(RelayPin, GPIO.HIGH)
    GPIO.cleanup()


def start_cam(camera_port, file,ramp_frames = 30):
    # Now we can initialize the camera capture object with the cv2.VideoCapture class.
    # All it needs is the index to a camera port.
    camera = cv2.VideoCapture(camera_port)
    # Ramp the camera - these frames will be discarded and are only used to allow v4l2
    # to adjust light levels, if necessary
    camera.set(3,1280)
    camera.set(4,960)

    temp = None
    for i in xrange(ramp_frames):
        temp = camera.read()[1]
    logging.debug ("Taking image...")
    # Take the actual image we want to keep
    camera_capture = temp
    
    # A nice feature of the imwrite method is that it will automatically choose the
    # correct format based on the file extension you provide. Convenient!
    cv2.imwrite(file, camera_capture)

    # You'll want to release the camera, otherwise you won't be able to create a new
    # capture object until your script exits
    del(camera)

def capture_images(fport,rport,tport):
    logging.info('Initializing Capture')
    image_path_dict_all = { fport:"",
                        rport:"",
                        tport:""}
    if led_enabled:
        act_led()
    for key in image_path_dict_all.keys():
        logging.debug ("reading: %s",key)
        if capture_from_camera:
            image_path_dict_all[key] = images_cam_folder_path + str(key) + '.jpg'
            if laptop_capture:
                image_path_dict_all[key] = images_cam_folder_path + str(0) + '.jpg'
                key = 0
            logging.debug ("Initializing Camera...%s",str(key))
            start_cam(key,image_path_dict_all[key])
        else:
            logging.debug ("Reading Locally Stored Images...")
            image_path_dict_all[key] = images_local_folder_path + str(key) + '.jpg'
            logging.debug ('image_path_dict_all %s',image_path_dict_all[key])
    #TODO: Validation Check
    if led_enabled:
        deact_led()
    return image_path_dict_all
