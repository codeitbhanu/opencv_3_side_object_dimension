from werkzeug.wrappers import Request, Response
from util_network import get_local_ip_address


@Request.application
def application(request):
    return Response('Hello Werkzeug !!')

def run_server(port=4000):
    server_ip = get_local_ip_address('enp0s26u1u1')
    from werkzeug.serving import run_simple
    run_simple(server_ip, port, application)

