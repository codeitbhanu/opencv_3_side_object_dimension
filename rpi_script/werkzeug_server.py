from werkzeug.wrappers import Request, Response
from util_network import get_local_ip_address

from disable_enable_print import *

@Request.application
def application(request):
    return Response('Hello Werkzeug !!')

def run_server(intf, port=4000):
    server_ip = get_local_ip_address(intf)
    from werkzeug.serving import run_simple
    run_simple(server_ip, port, application)

