import socket
import cv2
import numpy as np
import pickle


host= "192.168.99.91"#'localhost'
port= 15323
socket_own = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_own.connect((host, port))


first_color = cv2.imread("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/rgb/00000.png")
first_mask = cv2.imread("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/first_mask.png", cv2.IMREAD_GRAYSCALE)

send_data = dict()

send_data["mask"] = first_mask
send_data["rgb"] = first_color

send_pckl = pickle.dumps(send_data)
socket_own.sendall(send_pckl)

rec_data = b''
buffer_size = 4096
while True:
    part = socket_own.recv(buffer_size)
    rec_data += part
    if len(part) < buffer_size:
        # End of data (less than buffer_size means no more data left)
        break


print("waiting for packet...")
data = pickle.loads(rec_data)        


print(data)

color_inf_img = cv2.imread("/home/thws_robotik/Documents/Leyh/6dpose/datasets/BuchVideo/rgb/00001.png")

send_data["mask"] = None
send_data["rgb"] = color_inf_img

send_pckl = pickle.dumps(send_data)
socket_own.sendall(send_pckl)

rec_data = b''
buffer_size = 4096
while True:
    part = socket_own.recv(buffer_size)
    rec_data += part
    if len(part) < buffer_size:
        # End of data (less than buffer_size means no more data left)
        break

data = pickle.loads(rec_data)
mask = data["mask"]


cv2.imwrite("out/mask_pred.jpg", mask * 255)
print("Success!")
