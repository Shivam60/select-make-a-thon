import io,cv2
import socket,numpy as np
import struct
from PIL import Image

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
        #nparr =  np.array(image.getdata()).astype(np.float32).reshape( (image.size[0],image.size[1],3) )
        print(image)
#        cv2.imshow('Live',nparr)
#        cv2.waitKey(10)
        print('Image is %dx%d' % image.size)
        image.verify()
        print('Image is verified')
finally:
    connection.close()
    server_socket.close()

