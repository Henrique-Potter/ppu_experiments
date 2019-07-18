from coapthon.server.coap import CoAP
#from raspberry_pi.peyes_nc_resource import PeyesNC
import platform
import logging

logger = logging.getLogger()
logger.setLevel(level=logging.CRITICAL)

if platform.uname()[1] =='raspberrypi':
    from peyes_c_resource import PeyesC
    #from peyes_nc_resource import PeyesTrigger
else:
    from raspberry_pi.peyes_c_resource import PeyesC
    #from raspberry_pi.peyes_nc_resource import PeyesTrigger


class CoAPServer(CoAP):
    def __init__(self, host, port):
        CoAP.__init__(self, (host, port))

        self.add_resource('peyes/', PeyesC())
        #self.add_resource('peyes/', PeyesTrigger())


def main():

    server = CoAPServer("192.168.0.141", 5683)
    try:
        print("Server Started at {}".format(server.server_address))
        server.listen(10)

    except KeyboardInterrupt:
        print("Server Shutdown")
        server.close()
        print("Exiting...")
        if platform.uname()[1] =='raspberrypi':
            import RPi.GPIO as GPIO
            GPIO.cleanup()


if __name__ == '__main__':
    main()
