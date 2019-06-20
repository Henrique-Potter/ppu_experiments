import logging
import asyncio

from aiocoap import *

import numpy as np
from coapthon.client.helperclient import HelperClient

host = "localhost"
port = 5683
path_identify ="peyes"
path_learn ="peyes"


async def main():
    protocol = await Context.create_client_context()

    request = Message(code=GET, uri='coap://localhost/peyes')

    try:
        response = await protocol.request(request).response
    except Exception as e:
        print('Failed to fetch resource:')
        print(e)
    else:
        print('Result: %s\n%r'%(response.code, response.payload))


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

    client = HelperClient(server=(host, port))
    response = client.get(path_identify)
    print(response.pretty_print())

    # if np.any(response):
    #     image = np.fromstring(response, dtype=int)
    #
    # response = client.post(path_learn, "")
    # print(response.pretty_print())

    # response = client.get(path_identify)
    # print(response.pretty_print())

    client.stop()