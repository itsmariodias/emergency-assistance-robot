import RPi.GPIO as GPIO
from time import sleep

def SetAngle(angle, pwm, servo):
	duty = angle / 18 + 2
	GPIO.output(servo, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(servo, False)
	pwm.ChangeDutyCycle(0)

class Tray:
	
	def __init__(self, left=20, right=21):
		self.servo1 = right
		self.servo2 = left
		GPIO.setup(self.servo1, GPIO.OUT)
		GPIO.setup(self.servo2, GPIO.OUT)

		self.pwm1=GPIO.PWM(self.servo1, 50)
		self.pwm1.start(0)
		self.pwm2=GPIO.PWM(self.servo2, 50)
		self.pwm2.start(0)

	def OpenRightTray(self):
		SetAngle(90, self.pwm1, self.servo1)

	def CloseRightTray(self):
		SetAngle(0, self.pwm1, self.servo1)

	def OpenLeftTray(self):
		SetAngle(0, self.pwm2, self.servo2)

	def CloseLeftTray(self):
		SetAngle(90, self.pwm2, self.servo2)

	def __del__(self):
		self.pwm1.stop()
		self.pwm2.stop()
		GPIO.cleanup()
