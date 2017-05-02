import netifaces as ni
from disable_enable_print import *
def get_local_ip_address(interface_name):
    ni.ifaddresses(interface_name)
    ip = ni.ifaddresses(interface_name)[2][0]['addr']
    logging.debug ("Local IP Address: %s",ip)  # should logging.debug "192.168.100.37"
    return ip