import sys
from multiprocessing import Queue
sys.path.append("..")
from coapthon.resources.resource import Resource
from multiprocessing import Process
from pi_face_detection import PiFaceDet
from threading import Lock
import platform

peyes_lock = Lock()
inputQueue = Queue(maxsize=5)

continuous_server = False

beep_pin = 40
g_led_pin = 36
r_led_pin = 38


def start_face_det(learn_face_count):
    f = PiFaceDet(preview=False)
    f.continuous_face_detection(learn_face_count)


class PeyesC(Resource):

    def __init__(self, name="Peyes", coap_server=None):
        super(PeyesC, self).__init__(name,
                                     coap_server,
                                     visible=True,
                                     observable=True,
                                     allow_children=True)

        self.payload = "Peyes Continous Identification"
        self.max_age = 60

    def render_GET(self, request):
        print("[INFO] starting process...")

        global continuous_server

        if not continuous_server:
            self.beep_blink(3, r_led_pin, 0.5)
            p = Process(target=start_face_det, args=(inputQueue,))
            p.daemon = True
            p.start()
            continuous_server = True
            self.payload = "ID process started Successfully!"
        else:
            self.beep_blink(1, g_led_pin, 0.2)
            inputQueue.put(1)
            self.payload = "Trigger Sensor sent a get request!"
            self.payload = "Continuous ID process is already ON!"

        return self

    def render_GET_advanced(self, request, response):
        pass

    def render_PUT(self, request):
        self.payload = request.payload
        return self

    def render_POST(self, request):
        
        res = PeyesC()

        if continuous_server:
            inputQueue.put(2)
            inputQueue.put(2)
            inputQueue.put(2)
            inputQueue.put(2)
            inputQueue.put(2)
            res.payload = 'Request sent successfully'
        else:
            res.payload = 'ID server is not running'

        return res

    def render_DELETE(self, request):
        return True

    @staticmethod
    def beep_blink(blink_times, led_pin, duration=0.3):
        import RPi.GPIO as GPIO
        import time

        if platform.uname()[1] == 'raspberrypi':
            for i in range(blink_times):
                GPIO.output(beep_pin, GPIO.HIGH)
                GPIO.output(led_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(beep_pin, GPIO.LOW)
                GPIO.output(led_pin, GPIO.LOW)
                time.sleep(duration)

