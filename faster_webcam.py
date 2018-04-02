from threading import Thread, Lock
import cv2,os,numpy as np
face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt2.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
subjects = ["", "Ramiz Raja", "Elvis Presley","Shivam","NM"]
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
def predict(img):
    face, rect = detect_face(img)
    if face is None:
        print("face not detected, program exiting")
        return None
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)    
    return img
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if (len(faces) == 0):
        return None, None    
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]
def prepare_training_data(data_folder_path):    
    dirs = os.listdir(data_folder_path)   
    faces = []
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
            face, rect = detect_face(image)
            if face is None:
                print("face not detected, ignoring image",image_name)
            else:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels
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
#    print("Preparing data...")
#    faces, labels = prepare_training_data("training-data")
#    face_recognizer.train(faces, np.array(labels))
#    print("Data prepared")

    vs = WebcamVideoStream('http://192.168.225.222:8160').start()
    pv=None
    while True :
        frame = vs.read()
        cv2.imshow('v',frame)
        cv2.waitKey(1)
#        frame=predict(frame)
 #       if frame is not None:
 #           print(frame)
 #      
	    

    vs.stop()
    cv2.destroyAllWindows()
