import sys
sys.path.append("..")

from coapthon.resources.resource import Resource
from pi_face_detection import PiFaceDet
import RPi.GPIO as GPIO
from threading import Lock
import time


GPIO.setmode(GPIO.BOARD)

beep_pin = 40
g_led_pin = 36
r_led_pin = 38

GPIO.setup(beep_pin, GPIO.OUT, initial=0)
GPIO.setup(g_led_pin, GPIO.OUT, initial=0)
GPIO.setup(r_led_pin, GPIO.OUT, initial=0)

peyes_lock = Lock()

print("\n\n----------Tensorflow wam-up----------\n\n")
face_detection = PiFaceDet()
found_face = face_detection.run_identification(1)
print("\n\n----------Tensorflow wam-up complete----------\n\n")


class Peyes(Resource):

    def __init__(self, name="Peyes", coap_server=None):
        super(Peyes, self).__init__(name, coap_server, visible=True,
                                            observable=True, allow_children=True)

        self.payload = "Peyes"
        self.max_age = 60
        self.beep(1, 0.2)

    def render_GET(self, request):

        self.payload = "Failure"

        peyes_lock.acquire()
        print("get lock in")
        found_face = face_detection.run_identification(5)
        print("Face identified:{}".format(found_face))
        
        peyes_lock.release()
        print("get lock out")

        if found_face:
            self.payload = "True"
            self.green_blink(1, 2)
            self.beep(2)
        else:
            self.payload = "False"
            self.red_blink(1, 2)
            self.beep(1)

        return self

    def render_GET_advanced(self, request, response):
        pass

    def render_PUT(self, request):
        self.payload = request.payload
        return self

    def render_POST(self, request):
        
        res = Peyes()
        
        res.payload = "Failure"
        
        peyes_lock.acquire()
        print("post lock in")
        learn_face_status = face_detection.run_learn_face(5)
        peyes_lock.release()
        print("Face learned:{}".format(learn_face_status))
        print("post lock out")
        
        if learn_face_status:
            res.payload = "True"
            self.green_blink(3, 0.5)
            self.beep(3)
        else:
            res.payload = "False"
        
        return res

    def render_DELETE(self, request):
        return True

    @staticmethod
    def beep(beep_times, duration=0.1):
        for i in range(beep_times):
            GPIO.output(beep_pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(beep_pin, GPIO.LOW)
            time.sleep(duration)

    @staticmethod
    def red_blink(blink_times, duration=0.3):
        for i in range(blink_times):
            GPIO.output(r_led_pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(r_led_pin, GPIO.LOW)
            time.sleep(duration)

    @staticmethod
    def green_blink(blink_times, duration=0.3):
        for i in range(blink_times):
            GPIO.output(g_led_pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(g_led_pin, GPIO.LOW)
            time.sleep(duration)
