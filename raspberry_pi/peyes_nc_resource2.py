import sys
sys.path.append("..")

import aiocoap.resource as resource
import aiocoap
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


class PeyesTrigger(resource.Resource):
    """Example resource which supports the GET and PUT methods. It sends large
    responses, which trigger blockwise transfer."""

    def __init__(self):
        super().__init__()
        #self.set_content(b"This is the resource's default content. It is padded "\
         #       b"with numbers to be large enough to trigger blockwise "\
          #      b"transfer.\n")

    def set_content(self, content):
        self.content = content
        while len(self.content) <= 1024:
            self.content = self.content + b"0123456789\n"

    async def render_get(self, request):

        start1 = time.time()

        peyes_lock.acquire()

        found_face, frame_as_string = face_detection.id_face_trigger(1)
        peyes_lock.release()

        #self.payload = found_face

        print('Face ID time: {}'.format(time.time() - start1))

        self.content = frame_as_string.encode('utf16')
        #print(found_face)
        #print(self.payload)

        return aiocoap.Message(payload=self.content)

    async def render_put(self, request):
        print('PUT payload: %s' % request.payload)
        self.set_content(request.payload)
        return aiocoap.Message(code=aiocoap.CHANGED, payload=self.content)

#
#
# class PeyesTrigger(Resource):
#
#     def __init__(self, name="Peyes", coap_server=None):
#         super(PeyesTrigger, self).__init__(name, coap_server, visible=True,
#                                      observable=True, allow_children=True)
#         self.payload = "Peyes Trigger ID"
#         self.max_age = 60
#
#     def render_GET(self, request):
#         start1 = time.time()
#
#         peyes_lock.acquire()
#
#         found_face, frame_as_string = face_detection.id_face_trigger(1)
#         peyes_lock.release()
#
#         #self.payload = found_face
#
#         print('Face ID time: {}'.format(time.time() - start1))
#
#         string_list = ''.join(str(r) for v in frame_as_string for r in v)
#         self.payload = string_list
#         #print(found_face)
#         #print(self.payload)
#
#         return self
#
#     # def render_GET_advanced(self, request, response):
#     #     start1 = time.time()
#     #
#     #     peyes_lock.acquire()
#     #
#     #     found_face, frame_as_string = face_detection.id_face_trigger(1)
#     #     peyes_lock.release()
#     #
#     #     response.payload = self.payload = str(frame_as_string, 'UTF-8')
#     #
#     #     print('Face ID time: {}'.format(time.time() - start1))
#     #
#     #     print(frame_as_string)
#     #     print(self.payload)
#     #
#     #     return self, response
#
#     def render_PUT(self, request):
#         self.payload = request.payload
#         return self
#
#     def render_POST(self, request):
#
#         peyes_lock.acquire()
#         found_face = face_detection.learn_face_trigger()
#         peyes_lock.release()
#
#         res = PeyesTrigger()
#         res.payload = found_face
#
#         return res
#
#     def render_DELETE(self, request):
#         return True
