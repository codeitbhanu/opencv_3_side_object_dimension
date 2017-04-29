import sys, os

# Disable
def disable_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__

"""
print 'This will print'

disable_print()
print "This won't"

disable_print()
print "This will too"
"""