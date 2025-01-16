import requests

url = '127.0.0.1:5555/find/tengwar'
requests.post(url, files={'image': open('test/t1.png', 'rb')})