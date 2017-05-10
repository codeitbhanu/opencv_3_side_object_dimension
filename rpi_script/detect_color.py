import get_color as gc
from disable_enable_print import *
from webcolors import *
 
def get_color(img_dict, fport, rport, tport, profile=0):
    logging.debug ("Processing For Color...")
    # camera_port_front = 0 #img_front
    # camera_port_rear = 1 #img_rear
    # camera_port_top = 2 #img_top
    fr,fg,fb = gc.get(img_dict[fport], profile, imgType='front')
    fcode = str(rgb_to_hex((fr,fg,fb)))
    rr,rg,rb = gc.get(img_dict[rport], profile, imgType='rear')
    rcode = str(rgb_to_hex((rr,rg,rb)))

    return {'front':{'r':str(fr),
            'g':str(fg),
            'b':str(fb),
            'code':fcode
            },
            'rear':{'r':str(rr),
            'g':str(rg),
            'b':str(rb),
            'code':rcode
            }}
