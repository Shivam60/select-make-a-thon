from threading import Thread, Lock
import cv2
face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt2.xml')
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
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
#                cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
 #               cv2.waitKey(1)
#                draw_rectangle(face, rect)
            
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
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
    vs = WebcamVideoStream().start()
    pv=None
    while True :
        frame = vs.read()
        try:
            face,rect=detect_face(frame)
            draw_rectangle(frame,rect)
        except:
#            print(frame)
            print("face not detected")
        finally:
            cv2.imshow('v',frame)    
            cv2.waitKey(1)

    vs.stop()
    cv2.destroyAllWindows()
