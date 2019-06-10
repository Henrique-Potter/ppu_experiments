import sys
from multiprocessing import Queue
sys.path.append("..")
from coapthon.resources.resource import Resource
from multiprocessing import Process
from pi_c_face_detection import PiFaceDet
from threading import Lock

peyes_lock = Lock()
inputQueue = Queue(maxsize=3)

continuous_server = True


def start_face_det(learn_face_count):

    f = PiFaceDet()
    f.continuous_face_identification(learn_face_count)


class PeyesC(Resource):

    def __init__(self, name="Peyes", coap_server=None):
        super(PeyesC, self).__init__(name, coap_server, visible=True,
                                            observable=True, allow_children=True)

        self.payload = "Peyes Continous Identification"
        self.max_age = 60

    def render_GET(self, request):
        print("[INFO] starting process...")

        global continuous_server

        if continuous_server:
            p = Process(target=start_face_det, args=(inputQueue,))
            p.daemon = True
            p.start()
            continuous_server = False
            self.payload = "Success"
        else:
            self.payload = "Continuous ID is already ON"

        return self

    def render_GET_advanced(self, request, response):
        pass

    def render_PUT(self, request):
        self.payload = request.payload
        return self

    def render_POST(self, request):
        
        res = PeyesC()

        inputQueue.put(True)
        inputQueue.put(True)
        inputQueue.put(True)
        
        return res

    def render_DELETE(self, request):
        return True


