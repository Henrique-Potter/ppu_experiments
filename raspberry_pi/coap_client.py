import logging
import asyncio
import png
from aiocoap import *
import json
import numpy as np
from coapthon.client.helperclient import HelperClient
import cv2 as cv
import time

host = "192.168.0.141"
port = 5683
path_identify ="peyes"
path_learn ="peyes"


async def main():
    protocol = await Context.create_client_context()

    request = Message(code=GET)
    request.set_request_uri(uri='coap://[fe80::2518:6bb8:87de:5c45%12]/peyes')

    try:
        response = await protocol.request(request).response

        response_payload = response.payload.decode('utf16')
        response_data = json.loads(response_payload)

        face_frame_jpeg = np.asarray(response_data['detection_frame'], dtype='uint8')
        face_name = response_data['person_name']
        face_frame = cv.imdecode(face_frame_jpeg, 1)

        cv.imshow("Debugging", face_frame)
        cv.waitKey(0)
        cv.destroyAllWindows()


        # png.from_array(frame)
        # f = open('png.png', 'wb')
        # w = png.Writer(3, 2)
        # w.write(f, face_frame)
        # f.close()

    except Exception as e:
        print('Failed to fetch resource:')
        print(e)
    else:
        print('Result: %s\n%r'%(response.code, response.payload))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

    pass

    # client = HelperClient(server=(host, port))
    # response = client.get(path_identify)
    # print(response.pretty_print())
    #
    # # if np.any(response):
    # #     image = np.fromstring(response, dtype=int)
    # #
    # # response = client.post(path_learn, "")
    # # print(response.pretty_print())
    #
    # # response = client.get(path_identify)
    # # print(response.pretty_print())
    #
    # client.stop()