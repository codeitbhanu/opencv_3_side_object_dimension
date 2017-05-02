import cv2

capture_from_camera = True

laptop_capture = True

camera_port_front = 0 #img_front
camera_port_rear = 1 #img_rear
camera_port_top = 2 #img_top

images_local_folder_path = "img_local/"
images_cam_folder_path = "img_webcam/"

def start_cam(camera_port, file,ramp_frames = 30):
    # Now we can initialize the camera capture object with the cv2.VideoCapture class.
    # All it needs is the index to a camera port.
    camera = cv2.VideoCapture(camera_port)
    # Ramp the camera - these frames will be discarded and are only used to allow v4l2
    # to adjust light levels, if necessary
    temp = None
    for i in xrange(ramp_frames):
        temp = camera.read()[1]
    print("Taking image...")
    # Take the actual image we want to keep
    camera_capture = temp
    
    # A nice feature of the imwrite method is that it will automatically choose the
    # correct format based on the file extension you provide. Convenient!
    cv2.imwrite(file, camera_capture)

    # You'll want to release the camera, otherwise you won't be able to create a new
    # capture object until your script exits
    del(camera)

def capture_images():
    image_path_dict_all = { camera_port_front:"",
                        camera_port_top:"",
                        camera_port_rear:""}

    for key in image_path_dict_all.keys():
        print "reading: ",key
        if capture_from_camera:
            image_path_dict_all[key] = images_local_folder_path + str(key) + '.jpg'
            if laptop_capture:
                image_path_dict_all[key] = images_local_folder_path + str(0) + '.jpg'
                key = 0
            print "Initializing Camera..." + str(key)
            start_cam(key,image_path_dict_all[key])
        else:
            print "Reading Locally Stored Images..."
            image_path_dict_all[key] = images_local_folder_path + str(key) + '.jpg'
            print image_path_dict_all[key]
    #TODO: Validation Check
    return image_path_dict_all
