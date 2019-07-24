import sys
from multiprocessing import Queue
sys.path.append("..")
from coapthon.resources.resource import Resource
from multiprocessing import Process
from pi_face_detection import PiFaceDet
from threading import Lock
import pandas as pd
import platform
import time
import os

peyes_lock = Lock()
inputQueue = Queue(maxsize=5)

continuous_server = False

beep_pin = 40
g_led_pin = 36
r_led_pin = 38

counter = 10

if os.path.exists("trigger_metrics_2.csv"):
    trigger_metrics_2 = pd.read_csv("trigger_metrics_2.csv", index_col=False).values.tolist()
else:
    trigger_metrics_2 = []


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

        self.payload = "Peyes Continuous Identification"
        self.max_age = 60

    def render_GET(self, request):

        global continuous_server

        if not continuous_server:
            continuous_server = True
            print("[INFO] Get request received, starting Video Thread....")
            self.beep_blink(3, r_led_pin, 0.5)
            p = Process(target=start_face_det, args=(inputQueue,))
            p.daemon = True
            p.start()

            self.payload = "ID process started Successfully!"
        else:
            print("[INFO] Get request received as trigger...")
            self.beep_blink(2, g_led_pin, 0.1)
            inputQueue.put(time.time())
            self.payload = "Trigger Sensor sent a get request!"
            self.payload = "Continuous ID process is already ON!"
        return self

    def render_GET_advanced(self, request, response):
        pass

    def render_PUT(self, request):
        self.payload = request.payload
        return self

    def render_POST(self, request):

        global counter, trigger_metrics_2

        res = PeyesC()
        time_stamp = time.time()
        trigger_metrics_2.append([0, time_stamp])
        counter = counter - 1

        print('[INFO - TRIG 2] Get received at:{} Save deadline:{}'.format(time_stamp, counter))

        if counter < 1:
            counter = 10

            total_data_df = pd.DataFrame(trigger_metrics_2)
            try:
                total_data_df.to_csv("trigger_metrics_2.csv", index=False)
            except Exception as e:
                print(e)

            print('[INFO] Saving trigger metrics 2.')
            print(total_data_df)

        return res

    def render_DELETE(self, request):
        return True

    @staticmethod
    def beep_blink(blink_times, led_pin, duration=0.3):
        if platform.uname()[1] == 'raspberrypi':
            import RPi.GPIO as GPIO
            import time
            for i in range(blink_times):
                GPIO.output(beep_pin, GPIO.HIGH)
                GPIO.output(led_pin, GPIO.HIGH)
                time.sleep(duration)
                GPIO.output(beep_pin, GPIO.LOW)
                GPIO.output(led_pin, GPIO.LOW)
                time.sleep(duration)

