from threading import Thread, Lock
import pickle,cv2,os,numpy as np
from PIL import Image
import queue,io,cv2,struct,time,socket

cascade = 'opencv-files/lbpcascade_frontalface.xml'
names = ["Cannot Predict Name", "Jai Sagar", "VNAY","Shivam","Shreyam","Naman"]

q = queue.Queue(maxsize=0)
num_threads = 1
def do_stuff(q):
  while True:
      image=q.get()
      print(image.shape)
      if q.qsize()==0:
          print(0)
          time.sleep(.3)
      #frame=f.predict_face(image)
      #if frame == (None,None):
       #   frame=image
      cv2.imshow('picam',image)
      cv2.waitKey(1)
      print(q.qsize())
      q.task_done()

worker = Thread(target=do_stuff, args=(q,))
worker.setDaemon(True)
worker.start()

class FDP:
    def __init__(self,cascade,names,training_data='training-data',saved='trained'):
        self.face_cascade=cv2.CascadeClassifier(cascade)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names=names
        print("Learning from the Database")          
        if saved in os.listdir():
            faces, labels=self.load(saved)
        else:
            faces, labels =self.prepare_training_data(training_data)
            self.save("trained",pickle.dumps((faces,labels)))
        self.face_recognizer.train(faces, np.array(labels))
        print("Learning Finished")

    def save(self,name,stuff):
        fl= open(name,'wb')
        fl.write(stuff)
        fl.close()

    def load(self,name):
        fl=open(name,'rb')
        stuff=fl.read()
        return pickle.loads(stuff)    
    def predict_face(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if (len(faces) == 0):
            return None, None    
        (x, y, w, h) = faces[0] #face rectangle coordinates 
        face = gray[y:y+w, x:x+h] 
        name, confidence = self.face_recognizer.predict(face) #predict face
        #if confidence<99:
        #name=0
        name_text = self.names[0]
        #print(name)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #draw rectangle
        cv2.putText(img, name_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2) #draw text
        return img
    def prepare_training_data(self,data_folder_path):
  
        dirs = os.listdir(data_folder_path)   
        faces_dir = []
        labels = []    
        for dir_name in dirs:        
            if not dir_name.startswith("s"):
                continue            
            label = int(dir_name.replace("s", ""))
            subject_dir_path = data_folder_path + "/" + dir_name        
            subject_images_names = os.listdir(subject_dir_path)        
            for image_name in subject_images_names:            
                if image_name.startswith("."):
                    continue
                image_path = subject_dir_path + "/" + image_name
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                       
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                if (len(faces) != 0):
                    (x, y, w, h) = faces[0]
                    face=gray[y:y+w, x:x+h]
                    rect= faces[0]

                    faces_dir.append(face)
                    labels.append(label)        
                else:
                    print("face not detected, ignoring image",image_name)

        return faces_dir, labels

class WebcamVideoStream :
    def __init__(self, src = 0, width = 320, height = 240) :
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self
    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()
    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame
    def stop(self) :
        self.started = False
        self.thread.join()
    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()


if __name__ == "__main__" :
    f=FDP(cascade,names,'training-data')
    #vs =WebcamVideoStream().start()

    vs =WebcamVideoStream('http://192.168.225.222:8554').start()
    #server_socket = socket.socket()
    #server_socket.bind(('192.168.225.250', 8300))
    #server_socket.listen(1)
    #connection = server_socket.accept()[0].makefile('rb')
    #print('connection accepted')
    while True :
        image = vs.read()
        frame=f.predict_face(image)
#        q.put(image)
        if frame == (None,None):
            frame=image
        cv2.imshow('v',frame)
        cv2.waitKey(50)
'''
    while True:
        image_len = struct.unpack('<L', connection.read(4))[0]
        if not image_len:
            break
        image_stream = io.BytesIO()
        image_stream.write(connection.read(image_len))
        image_stream.seek(0)
        image = Image.open(image_stream)        
        q.put(image)
'''    

