#Here is the configuration file to read lowHSV and highHSV for required views

CONFIG_FILE = 'rpi_script/config.json'

#IMPORTS SYS PACKAGES 
import json
from pathlib import Path

#IMPORTS LOCAL PACKAGES
from disable_enable_print import *


json_file = Path(CONFIG_FILE)

config_data = {}

def process_config():
    if not json_file.is_file():
        logging.debug("ERROR: Unable to locate config in this path: %s"%(CONFIG_FILE))
        return
    
    with open(CONFIG_FILE) as in_file:
        global config_data
        config_data = json.load(in_file)
        # print data['type2']['img_proc']['front']
        in_file.close()
        return config_data
        
def write_config():
    if not json_file.is_file():
        logging.debug("NO EXISTING CONFIG FOUND, WRITING NEW CONFIG %s"%(CONFIG_FILE))
        config = {}
        ### TYPE_1:BEGIN ###
        config['type1'] = {}
        ### TYPE_1_IMG_PROC ###
        config['type1']['img_proc'] = {}
        config['type1']['img_proc']['front'] = {}
        config['type1']['img_proc']['front']['lowH'] = 0
        config['type1']['img_proc']['front']['lowS'] = 0
        config['type1']['img_proc']['front']['lowV'] = 0
        
        config['type1']['img_proc']['front']['highH'] = 180
        config['type1']['img_proc']['front']['highS'] = 140
        config['type1']['img_proc']['front']['highV'] = 256
        
        config['type1']['img_proc']['top'] = {}
        config['type1']['img_proc']['top']['lowH'] = 0
        config['type1']['img_proc']['top']['lowS'] = 0
        config['type1']['img_proc']['top']['lowV'] = 0
        
        config['type1']['img_proc']['top']['highH'] = 104
        config['type1']['img_proc']['top']['highS'] = 120
        config['type1']['img_proc']['top']['highV'] = 180
        
        ### TYPE_1_GET_COLOR ###
        config['type1']['get_color'] = {}
        config['type1']['get_color']['front'] = {}
        config['type1']['get_color']['front']['lowH'] = 0
        config['type1']['get_color']['front']['lowS'] = 0
        config['type1']['get_color']['front']['lowV'] = 0
        
        config['type1']['get_color']['front']['highH'] = 134
        config['type1']['get_color']['front']['highS'] = 154
        config['type1']['get_color']['front']['highV'] = 180
        
        config['type1']['get_color']['rear'] = {}
        config['type1']['get_color']['rear']['lowH'] = 0
        config['type1']['get_color']['rear']['lowS'] = 0
        config['type1']['get_color']['rear']['lowV'] = 0
        
        config['type1']['get_color']['rear']['highH'] = 193
        config['type1']['get_color']['rear']['highS'] = 169
        config['type1']['get_color']['rear']['highV'] = 240
        ### TYPE_1:ENDS ###

        ### TYPE_2:BEGIN ###
        config['type2'] = {}
        ### TYPE_2_IMG_PROC ###
        config['type2']['img_proc'] = {}
        config['type2']['img_proc']['front'] = {}
        config['type2']['img_proc']['front']['lowH'] = 0
        config['type2']['img_proc']['front']['lowS'] = 0
        config['type2']['img_proc']['front']['lowV'] = 0
        
        config['type2']['img_proc']['front']['highH'] = 180
        config['type2']['img_proc']['front']['highS'] = 140
        config['type2']['img_proc']['front']['highV'] = 256
        
        config['type2']['img_proc']['top'] = {}
        config['type2']['img_proc']['top']['lowH'] = 0
        config['type2']['img_proc']['top']['lowS'] = 0
        config['type2']['img_proc']['top']['lowV'] = 0
        
        config['type2']['img_proc']['top']['highH'] = 104
        config['type2']['img_proc']['top']['highS'] = 120
        config['type2']['img_proc']['top']['highV'] = 180
        
        ### TYPE_2_GET_COLOR ###
        config['type2']['get_color'] = {}
        config['type2']['get_color']['front'] = {}
        config['type2']['get_color']['front']['lowH'] = 0
        config['type2']['get_color']['front']['lowS'] = 0
        config['type2']['get_color']['front']['lowV'] = 0
        
        config['type2']['get_color']['front']['highH'] = 134
        config['type2']['get_color']['front']['highS'] = 154
        config['type2']['get_color']['front']['highV'] = 180
        
        config['type2']['get_color']['rear'] = {}
        config['type2']['get_color']['rear']['lowH'] = 0
        config['type2']['get_color']['rear']['lowS'] = 0
        config['type2']['get_color']['rear']['lowV'] = 0
        
        config['type2']['get_color']['rear']['highH'] = 193
        config['type2']['get_color']['rear']['highS'] = 169
        config['type2']['get_color']['rear']['highV'] = 240
        ### TYPE_2:ENDS ###

        with open(CONFIG_FILE,'w') as out_file:
            json.dump(config,out_file)
            out_file.close();

    else:
        logging.debug("CONFIGURATION ALREADY EXIST KEEP IT OR LEAVE IT, DECIDE NEXT TIME, I WILL USE EXISTING ONE...%s"%(CONFIG_FILE))



"""
        glob_lowH = 0
        glob_highH = 240

        # glob_lowS = 10
        glob_lowS = 0
        glob_highS = 180

        glob_lowV = 44
        glob_highV = 256
"""

#MAIN IS MAIN
if __name__ == '__main__':
    process_config()
    # write_config()
    print config_data['type2']