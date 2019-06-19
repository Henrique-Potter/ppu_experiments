import sys
sys.path.append("..")

from coapthon.resources.resource import Resource
from pi_face_detection import PiFaceDet
from threading import Lock
import time

peyes_lock = Lock()


print("\n\n----------Tensorflow Loading----------\n\n")
face_detection = PiFaceDet()
print("\n----------Tensorflow Loading complete----------\n")
print("\n\n----------Tensorflow wam-up----------\n\n")
face_detection.id_face_trigger(1)
print("\n\n----------Tensorflow wam-up complete----------\n\n")


class PeyesTrigger(Resource):

    def __init__(self, name="Peyes", coap_server=None):
        super(PeyesTrigger, self).__init__(name, coap_server, visible=True,
                                     observable=True, allow_children=True)
        self.payload = "Peyes Trigger ID"
        self.max_age = 60

    def render_GET(self, request):
        start1 = time.time()

        peyes_lock.acquire()

        found_face, frame_as_string = face_detection.id_face_trigger(1)
        peyes_lock.release()

        #self.payload = found_face

        print('Face ID time: {}'.format(time.time() - start1))

        #print(found_face)
        #print(self.payload)

        return self

    # def render_GET_advanced(self, request, response):
    #     start1 = time.time()
    #
    #     peyes_lock.acquire()
    #
    #     found_face, frame_as_string = face_detection.id_face_trigger(1)
    #     peyes_lock.release()
    #
    #     response.payload = self.payload = str(frame_as_string, 'UTF-8')
    #
    #     print('Face ID time: {}'.format(time.time() - start1))
    #
    #     print(frame_as_string)
    #     print(self.payload)
    #
    #     return self, response

    def render_PUT(self, request):
        self.payload = request.payload
        return self

    def render_POST(self, request):

        peyes_lock.acquire()
        found_face = face_detection.learn_face_trigger()
        peyes_lock.release()

        res = PeyesTrigger()
        res.payload = found_face

        return res

    def render_DELETE(self, request):
        return True
