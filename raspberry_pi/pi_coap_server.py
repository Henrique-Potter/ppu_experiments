from coapthon.server.coap import CoAP
from peyes_resource import Peyes
import RPi.GPIO as GPIO

class CoAPServer(CoAP):
    def __init__(self, host, port):
        CoAP.__init__(self, (host, port))

        self.add_resource('peyes/', Peyes())


def main():
    server = CoAPServer("192.168.0.141", 5683)
    try:
        print("Server Started at {}".format(server.server_address))
        server.listen(30)

    except KeyboardInterrupt:
        print("Server Shutdown")
        GPIO.cleanup()
        server.close()
        print("Exiting...")


if __name__ == '__main__':
    main()
