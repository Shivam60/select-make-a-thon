import io,cv2
import socket,numpy as np
import struct,time
from PIL import Image

from threading import Thread
import queue
#video=cv2.VideoWriter('video.avi',-1,1,(width,height))
def do_stuff(q):
  while True:
  	image=q.get()
  	cv2.imshow('picam',np.array(image))
  	cv2.waitKey(1)
  	#print()
  	#nparr =  np.array(image.getdata()).astype(np.float32).reshape( (image.size[0],image.size[1],3))
#  	print(image)
  	q.task_done()
		

q = queue.Queue(maxsize=0)
num_threads = 1

for i in range(num_threads):
  worker = Thread(target=do_stuff, args=(q,))
  worker.setDaemon(True)
  worker.start()
  
server_socket = socket.socket()
server_socket.bind(('169.254.50.99', 8000))
server_socket.listen(0)


connection = server_socket.accept()[0].makefile('rb')
try:
    while True:
        # Read the length of the image as a 32-bit unsigned int. If the
        # length is zero, quit the loop
        image_len = struct.unpack('<L', connection.read(4))[0]
        if not image_len:
            break
        # Construct a stream to hold the image data and read the image
        # data from the connection
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        # Rewind the stream, open it as an image with PIL and do some
        # processing on it
        image_stream.seek(0)
        image = Image.open(image_stream)
        
        q.put(image)
#        cv2.imshow('Live',nparr)
#        cv2.waitKey(10)
#        print('Image is %dx%d' % image.size)
#        image.verify()
#        print('Image is verified')
finally:
    connection.close()
    server_socket.close()

