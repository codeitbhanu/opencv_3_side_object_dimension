import netifaces as ni
def get_local_ip_address(interface_name):
    ni.ifaddresses(interface_name)
    ip = ni.ifaddresses(interface_name)[2][0]['addr']
    print "Local IP Address: ",ip  # should print "192.168.100.37"
    return ip