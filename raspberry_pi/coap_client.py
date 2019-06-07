from coapthon.client.helperclient import HelperClient

host = "192.168.0.141"
port = 5683
path_identify ="peyes"
path_learn ="peyes"

client = HelperClient(server=(host, port))
response = client.get(path_identify)
print(response.pretty_print())

response = client.post(path_learn, "")
print(response.pretty_print())

# response = client.get(path_identify)
# print(response.pretty_print())

client.stop()