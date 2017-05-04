from util_network import get_local_ip_address
from flask import Flask, redirect, url_for, request

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
import capture
import jsonify
import process_images
clr_profile = 0
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
from disable_enable_print import *
"""
app = Flask(__name__)

__app_name__ = 'run'

@app.route('/'+__app_name__+'/')
def application():
    img_dict = capture.capture_images()
    logging.debug (img_dict)

    ret = process_images.process(img_dict,clr_profile)
    # TODO: Return Type to Decide, dimension per image, Color Per Image
    # (w,h,l,#FFEEDD)

    enable_print()
    logging.info ("Processed Data: %s",ret)

    ### return as json
    # retJSON = json.dumps(ret)
    return str(dict)
app.add_url_rule('/',__app_name__,application)
"""
from flask import Flask, redirect, url_for, request
app = Flask(__name__)


@app.route('/execute/<name>')
def execute(belt):
    logging.info('*** Started Executing For Belt: %s ***',belt)
    img_dict = capture.capture_images()
    logging.debug(img_dict)

    ret = process_images.process(img_dict, clr_profile)
    # TODO: Return Type to Decide, dimension per image, Color Per Image
    # (w,h,l,#FFEEDD)

    enable_print()
    logging.info("Processed Data: %s", ret)

    # return as json
    # return jsonify(results=ret,ensure_ascii=False)
    # return 'welcome %s' % name
    return str(dict)


@app.route('/run', methods=['POST', 'GET'])



def run():
    if request.method == 'POST':
        belt = request.form['belt']
        # return redirect(url_for('execute', name=user))
        return execute(belt)
    else:
        belt = request.args.get('belt')
        # return redirect(url_for('execute', name=user))
        return execute(belt)


if __name__ == '__main__':
    intf = 'enp0s20u3'
    port = 5000
    server_ip = get_local_ip_address(intf)    
    app.run(server_ip,port,debug=True)