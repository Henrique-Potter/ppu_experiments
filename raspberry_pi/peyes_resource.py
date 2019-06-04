from coapthon.resources.resource import Resource
from pi_face_detection import PiFaceDet


class Peyes(Resource):

    def __init__(self, name="Peyes", coap_server=None):
        super(Peyes, self).__init__(name, coap_server, visible=True,
                                            observable=True, allow_children=True)

        self.face_detection = PiFaceDet()
        self.payload = "Basic Resource"

    def render_GET(self, request):
        return self

    def render_GET_advanced(self, request, response):
        pass

    def render_PUT(self, request):
        self.payload = request.payload
        return self

    def render_POST(self, request):
        res = Peyes()
        res.location_query = request.uri_query
        res.payload = request.payload
        return res

    def render_DELETE(self, request):
        return True
