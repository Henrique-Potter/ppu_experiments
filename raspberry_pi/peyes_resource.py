from coapthon.resources.resource import Resource
from pi_face_detection import PiFaceDet

face_detection = PiFaceDet()
from threading import Lock

peyes_lock = Lock()


class Peyes(Resource):

    def __init__(self, name="Peyes", coap_server=None):
        super(Peyes, self).__init__(name, coap_server, visible=True,
                                            observable=True, allow_children=True)

        self.payload = "Peyes"
        self.max_age = 60

    def render_GET(self, request):

        self.payload = "Failure"

        peyes_lock.acquire()
        found_face = face_detection.run_identification(50)
        peyes_lock.release()

        if found_face:
            self.payload = "True"
        else:
            self.payload = "False"

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
        learn_face_status = face_detection.run_learn_face(50)
        peyes_lock.release()

        if learn_face_status:
            res.payload = "True"
        else:
            res.payload = "False"

        return res

    def render_DELETE(self, request):
        return True
