from multiprocessing import Process

from coapthon.server.coap import CoAP
#from raspberry_pi.peyes_nc_resource import PeyesNC
from raspberry_pi.peyes_c_resource import PeyesC
import platform


class CoAPServer(CoAP):
    def __init__(self, host, port):
        CoAP.__init__(self, (host, port))

        self.add_resource('peyes/', PeyesC())
     #   self.add_resource('peyes_continuous/', PeyesNC())


def main():

    server = CoAPServer("192.168.0.177", 5683)
    try:
        print("Server Started at {}".format(server.server_address))
        server.listen(30)

    except KeyboardInterrupt:
        print("Server Shutdown")
        server.close()
        print("Exiting...")
        if platform.uname()[1] is 'raspberrypi':
            import RPi.GPIO as GPIO
            GPIO.cleanup()


if __name__ == '__main__':
    main()
