from util_network import get_local_ip_address
from flask import Flask, redirect, url_for, request

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
import capture
from flask import jsonify
import json
import process_images
clr_profile = 0
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
from disable_enable_print import *
app = Flask(__name__)


@app.route('/')
def root():
    return 'It works!!'


@app.route('/execute/<name>')
def execute(belt):
    logging.info('*** Started Executing For Belt: %s ***', belt)
    img_dict = capture.capture_images()
    logging.debug(img_dict)

    ret = process_images.process(img_dict, clr_profile)
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
        logging.debug('##### BELT: %s #####', belt)
        # return redirect(url_for('execute', name=user))
        return execute(belt)
    else:
        belt = request.args.get('belt')
        logging.info('##### BELT: %s #####', belt)
        # return redirect(url_for('execute', name=user))
        return execute(belt)


onLocalHost = False
if __name__ == '__main__':
    if onLocalHost == False:
        intf = 'enp0s26u1u1'
        port = 5000
        server_ip = get_local_ip_address(intf)
        app.run(server_ip,port,debug=True)
    else:
        app.run(debug=True)
