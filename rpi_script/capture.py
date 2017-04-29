capture_from_camera = False

camera_port_front = 0 #img_front
camera_port_rear = 1 #img_rear
camera_port_top = 2 #img_top

images_local_folder_path = "img_local/"
images_cam_folder_path = "img_webcam/"

def capture_images():
    image_path_dict_all = { camera_port_front:"",
                        camera_port_top:"",
                        camera_port_rear:""}

    for key in image_path_dict_all.keys():
        print "reading: ",key
        if capture_from_camera:
            print "Initializing Camera..."
            #TODO: write logic to capture and close camera, save to images_cam_folder_path
            print "No Camera Attached."
        else:
            print "Reading Locally Stored Images..."
            image_path_dict_all[key] = images_local_folder_path + str(key) + '.jpg'
            print image_path_dict_all[key]
    #TODO: Validation Check
    return image_path_dict_all
