import RPi.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(18, GPIO.OUT, initial=GPIO.LOW)
while True:
	GPIO.output(16, GPIO.HIGH)
	sleep(1)
	GPIO.output(16, GPIO.LOW)
	sleep(1)
	GPIO.output(18, GPIO.HIGH)
	sleep(1)
	GPIO.output(18, GPIO.LOW)
	sleep(1)