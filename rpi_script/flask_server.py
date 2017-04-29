from util_network import get_local_ip_address
from flask import Flask, redirect, url_for, request
app = Flask(__name__)

__app_name__ = 'myWebApp'

@app.route('/'+__app_name__+'/')
def application():
    return 'Hello Flasdfssdfdsffdsk !!!'
app.add_url_rule('/',__app_name__,application)

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))

active_opt = 0
def run_server(intf, port=5000):
    server_ip = get_local_ip_address(intf)    
    app.run(server_ip,port,debug=True)


"""
if __name__ == '__main__':
   app.run(debug = True)
"""