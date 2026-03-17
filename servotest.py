import RPi.GPIO as GPIO
import time

servo_pin = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)

pwm = GPIO.PWM(servo_pin, 50)   # 50Hz for servo
pwm.start(0)

try:
    while True:
        pwm.ChangeDutyCycle(2.5)  # 0 degree
        time.sleep(1)

        #pwm.ChangeDutyCycle(7.5)  # 90 degree
        #time.sleep(1)

        #pwm.ChangeDutyCycle(12) # 180 degree
        #time.sleep(2)
        pwm.ChangeDutyCycle(4.5) #45 degree
        time.sleep(1)

        pwm.ChangeDutyCycle(10) #135 degree
        time.sleep(1)

except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()
