def move_servo(duty):
    print("start moving servo to ", duty)
    set_servo_angle(duty)
    print("done moving servo to ", duty)


def set_servo_angle(duty):
    print("moving to ", duty)
