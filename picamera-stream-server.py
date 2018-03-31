import queue,io,cv2,struct,time,socket,numpy as np
from PIL import Image
from threading import Thread

def do_stuff(q):
  while True:
  	image=q.get()
  	cv2.imshow('picam',np.array(image))
  	cv2.waitKey(1)
  	q.task_done()
		
q = queue.Queue(maxsize=0)
num_threads = 1

worker = Thread(target=do_stuff, args=(q,))
worker.setDaemon(True)
worker.start()
  
server_socket = socket.socket()
server_socket.bind(('169.254.50.99', 8000))
server_socket.listen(0)


connection = server_socket.accept()[0].makefile('rb')
try:
    while True:
        image_len = struct.unpack('<L', connection.read(4))[0]
        if not image_len:
            break
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream)        
        q.put(image)
finally:
    connection.close()
    server_socket.close()

