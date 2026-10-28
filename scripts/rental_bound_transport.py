import argparse
import os
import socket
import sys
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--interface', required=True)
args = parser.parse_args()
if sys.platform != 'darwin':
    parser.error('This transport requires the macOS IP_BOUND_IF socket option')
connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
connection.setsockopt(socket.IPPROTO_IP, 25, socket.if_nametoindex(args.interface))
connection.settimeout(12)
connection.connect(('117.50.181.105', 29410))
connection.settimeout(None)

def upload():
    try:
        while True:
            block = os.read(0, 262144)
            if not block:
                connection.shutdown(socket.SHUT_WR)
                return
            connection.sendall(block)
    except OSError as error:
        print(type(error).__name__, file=sys.stderr, flush=True)
        os._exit(1)

threading.Thread(target=upload, daemon=True).start()
try:
    while True:
        block = connection.recv(262144)
        if not block:
            break
        view = memoryview(block)
        while view:
            written = os.write(1, view)
            view = view[written:]
finally:
    connection.close()
os._exit(0)
