import socket,struct,threading,picamera,io,time

client_socket = socket.socket()
client_socket.connect(('192.168.225.250', 8300))
connection = client_socket.makefile('wb')
try:
    connection_lock = threading.Lock()
    pool = []
    pool_lock = threading.Lock()

    class ImageStreamer(threading.Thread):
        def __init__(self):
            super(ImageStreamer, self).__init__()
            self.stream = io.BytesIO()
            self.event = threading.Event()
            self.terminated = False
            self.start()

        def run(self):
            # This method runs in a background thread
            while not self.terminated:
                if self.event.wait(1):
                    try:
                        with connection_lock:
                            connection.write(struct.pack('<L', self.stream.tell()))
                            connection.flush()
                            self.stream.seek(0)
                            connection.write(self.stream.read())
                    finally:
                        self.stream.seek(0)
                        self.stream.truncate()
                        self.event.clear()
                        with pool_lock:
                            pool.append(self)

    count = 0
    start = time.time()
    finish = time.time()

    def streams():
        global count, finish
        while finish - start < 30:
            with pool_lock:
                streamer = pool.pop()
            yield streamer.stream
            streamer.event.set()
            count += 1
            finish = time.time()

    with picamera.PiCamera() as camera:
        pool = [ImageStreamer() for i in range(4)]
        camera.resolution = (100, 50)
        camera.framerate = 12
        camera.start_preview()
        time.sleep(4)
        camera.capture_sequence(streams(), 'jpeg', use_video_port=True)

    while pool:
        with pool_lock:
            streamer = pool.pop()
        streamer.terminated = True
        streamer.join()

    with connection_lock:
        connection.write(struct.pack('<L', 0))

finally:
    connection.close()
    client_socket.close()

print('Sent %d images in %.2f seconds at %.2ffps' % (
    count, finish-start, count / (finish-start)))
