#/usr/bin/python
import RPi.GPIO as GPIO
import argparse
from time import sleep

RelayPin = 11

def act_led():
    GPIO.setmode(GPIO.BOARD) # Set GPIO as numbering
    GPIO.setup(RelayPin, GPIO.OUT)
    GPIO.output(RelayPin, GPIO.LOW)

def deact_led():
    GPIO.output(RelayPin, GPIO.HIGH)
    GPIO.cleanup()

if __name__ == '__main__':
    
    # TODO:
    # For Loop 3 Camera Run

    ap = argparse.ArgumentParser()
    ap.add_argument("opt", "--option", required=True,
                    help="Usage: exe opt [on/off/pulse]")
    args = vars(ap.parse_args())

    try:
        if args['option'] == 'on':
            act_led()
        if args['option'] == 'off':
            deact_led()
        if args['option'] == 'pulse':
            while True:
                act_led()
                sleep(1)
                deact_led()
                sleep(1)
        else:
            print "Usage: exe opt [on/off/pulse]"
    except KeyboardInterrupt:
        print 'De-activating LEDs'
        deact_led()