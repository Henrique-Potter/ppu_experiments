import sys
sys.path.append("..")

from coapthon.resources.resource import Resource
from pi_face_detection import PiFaceDet
import RPi.GPIO as GPIO
from threading import Lock
import time

face_detection = PiFaceDet()
GPIO.setmode(GPIO.BOARD)
mypin = 40
GPIO.setup(mypin, GPIO.OUT, initial=0)

peyes_lock = Lock()
print("\n\n----------Tensorflow wam-up----------\n\n")
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
            self.beep(2)
        else:
            self.payload = "False"
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
            self.beep(3)
        else:
            res.payload = "False"
        
        return res

    def render_DELETE(self, request):
        return True

    @staticmethod
    def beep(beep_times, duration=0.1):
        for i in range(beep_times):
            GPIO.output(mypin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(mypin, GPIO.LOW)
            time.sleep(duration)

