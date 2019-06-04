from coapthon.server.coap import CoAP
from raspberry_pi.peyes_resource import Peyes


class CoAPServer(CoAP):
    def __init__(self, host, port):
        CoAP.__init__(self, (host, port))
        self.add_resource('peyes/', Peyes())


def main():
    server = CoAPServer("192.168.0.177", 5683)
    try:
        server.listen(10)
    except KeyboardInterrupt:
        print("Server Shutdown")
        server.close()
        print("Exiting...")


if __name__ == '__main__':
    main()
