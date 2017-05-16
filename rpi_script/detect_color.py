import get_color as gc
from disable_enable_print import *
from webcolors import *
 
def get_color(img_dict, config_data, fport, rport, tport, profile=0):
    logging.debug ("Processing For Color...")

    fr,fg,fb = gc.get(img_dict[fport], config_data, profile, imgType='front')
    fcode = str(rgb_to_hex((fr,fg,fb)))
    rr,rg,rb = gc.get(img_dict[rport], config_data, profile, imgType='rear')
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
    """
    return {'front':{'r':str(143),
            'g':str(130),
            'b':str(121),
            'code':'#8f8279'
            },
            'rear':{'r':str(143),
            'g':str(140),
            'b':str(141),
            'code':'#8f8279'
            }}
    """