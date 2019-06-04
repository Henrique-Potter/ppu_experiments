from coapthon.client.helperclient import HelperClient

host = "192.168.0.177"
port = 5683
path_identify ="peyes?identify"
path_learn ="peyes?learn_face"

client = HelperClient(server=(host, port))
response = client.get(path_identify)
print(response.pretty_print())

response = client.get(path_learn)
print(response.pretty_print())

response = client.get(path_identify)
print(response.pretty_print())

client.stop()