# Save this as client.py on your Windows machine
import socket
import random

# Set up the client socket


def send(x,y,z):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = '192.168.2.3'  # The server's IP address
    port = 12345
    client_socket.connect((host, port))

    try:
        pose = f"Pose(x={x}, y={y}, z={z})"
        
        # Send the pose data
        client_socket.sendall(pose.encode('utf-8'))
        print('Sent:', pose)
    finally:
        client_socket.close()
