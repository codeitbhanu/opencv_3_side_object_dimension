import sys, os, logging
from disable_enable_print import *

logWriteType = 'console' #file, console,
logFileName = 'logging/logfile.log'

d = {'belt':'107','file':'tempfile'}

LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARN,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}
#[%s(filename)s:%(lineno)20s - %(funcName)20s()]
FORMAT = '[%(asctime)-15s] [%(levelname)s] [%(filename)s:%(lineno)s]\t %(message)s'


logging_level_name = 'info'
logging_level = LEVELS.get(logging_level_name, logging.NOTSET)

if logWriteType == 'file':
    logging.basicConfig(filename=logFileName,level=logging_level, format=FORMAT)
elif logWriteType == 'console':
    logging.basicConfig(level=logging_level, format=FORMAT)
else:
    print ("TODO...")


# logging.debug('Aur bhai debug karra h???')
# logging.info('Aur bhai info karra h???')
# logging.warning('Aur bhai warning karra h???')
# logging.error('Aur bhai error karra h???')
# logging.critical('Aur bhai critical karra h???')


# Disable
def disable_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__
