from util_network import get_local_ip_address
from flask import Flask
app = Flask(__name__)

@app.route('/')
def application():
    return 'Hello Flasdfssdfdsffdsk !!!'

def run_server(port=5000):
    server_ip = get_local_ip_address('enp0s26u1u1')
    app.run(server_ip,port,debug=True)