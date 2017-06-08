from util_network import get_local_ip_address
from flask import Flask, redirect, url_for, request

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
import capture
from flask import jsonify
# from flask.ext.cors import CORS, cross_origin
from flask_cors import CORS, cross_origin
#https://stackoverflow.com/questions/25594893/how-to-enable-cors-in-flask-and-heroku
import json
import process_images
import platform

camera_port_front = 2 #img_front       #DEBUG_TRY
camera_port_rear = 1 #img_rear
camera_port_top = 0 #img_top


clr_profile = 0
global config_data
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
from disable_enable_print import *
from configuration import *


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
    
@cross_origin()
@app.route('/')
def root():
    return 'It works!!'

@app.route('/execute/<name>')
def execute(belt, biscuit):
    global config_data
    logging.info('*** Started Executing For Belt: %s ***', belt)
    logging.info('*** Started Executing For Biscuit: %s ***', biscuit)
    config_data = process_config()
    config_data = config_data['type1']
    # logging.debug ("Config Data: %s"%(config_data['type1'])
    
    img_dict = capture.capture_images(camera_port_front,camera_port_rear,camera_port_top)
    logging.debug (img_dict)

    ret = process_images.process(img_dict, config_data, camera_port_front,camera_port_rear,camera_port_top,clr_profile)
    # TODO: Return Type to Decide, dimension per image, Color Per Image
    # (w,h,l,#FFEEDD)

    enable_print()
    logging.info("Processed Data: %s", ret)
    # return str(ret)
    # ret = jsonify(ret)
    # ret = json.dumps(ret)
    ###############################

    try:
        # json_dict = request.get_json()
        # stripeAmount = json_dict['stripeAmount']
        # stripeCurrency = json_dict['stripeCurrency']
        # stripeToken = json_dict['stripeToken']
        # stripeDescription = json_dict['stripeDescription']
        actualData = ret
        data = json.dumps(ret)
        # data = {'color_front': "stripeAmountRet", 'stripeCurrencyRet': "stripeCurrency", 'stripeTokenRet': "stripeToken", 'stripeDescriptionRet': "stripeDescription"}

        return data
    except Exception as e:
        logging.error("Error while returning data : " + str(e.args))
        return "Error in Server Return Data"


@app.route('/run', methods=['POST', 'GET'])
def run():
    logging.info('Got Call From Client')
    if request.method == 'POST':
        belt = request.form['belt']
        biscuit = request.form['biscuit']
        logging.debug('##### BELT: %s #####', belt)
        logging.debug('##### Biscuit: %s #####', biscuit)
        # return redirect(url_for('execute', name=user))
        return execute(belt,biscuit)
    else:
        logging.debug(request.args);
        belt = request.args.get('belt')
        biscuit = request.args.get('biscuit')
        logging.debug('##### BELT: %s #####', belt)
        logging.debug('##### Biscuit: %s #####', biscuit)
        # return redirect(url_for('execute', name=user))
        return execute(belt,biscuit)


PLATFORM_MACHINE = platform.machine()

onLocalHost = False
print PLATFORM_MACHINE
if PLATFORM_MACHINE is not 'armv7l':
    onLocalHost = True

if __name__ == '__main__':
    if onLocalHost == False:
        intf = 'wlan0'
        port = 5000
        server_ip = get_local_ip_address(intf)
        app.run(server_ip,port,debug=True)
    else:
        app.run(debug=True)
