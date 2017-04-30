import get_color_front as gcf
import get_color_rear as gcr
 
def get_color(img_dict, profile=0):
    print "Processing For Color..."
    # camera_port_front = 0 #img_front
    # camera_port_rear = 1 #img_rear
    # camera_port_top = 2 #img_top
    fr,fg,fb = gcf.get(img_dict[0])
    rr,rg,rb = gcr.get(img_dict[1])

    return {'front':{'r':fr,
            'g':fg,
            'b':fb
            },
            'rear':{'r':rr,
            'g':rg,
            'b':rb
            }}