import cv2,os
import numpy as np

face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt2.xml')
subjects = ["", "Ramiz Raja", "Elvis Presley","Shivam","NM"]
#classifiers=['opencv-files/haarcascade_frontalface_alt.xml', 'opencv-files/lbpcascade_frontalface.xml', 'opencv-files/haarcascade_profileface.xml', 'opencv-files/haarcascade_frontalface_alt.xml', 'opencv-files/lbpcascade_profileface.xml']
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

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
                print("face not detected, program exiting",image_name)
            else:
                cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                cv2.waitKey(1)

                draw_rectangle(face, rect)
            
            #------STEP-4--------
            #for the purpose of this tutorial
            #we will ignore faces that are not detected
            if face is not None:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


'''
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))
'''


def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)
    if face is None:
        print("face not detected, program exiting")
        exit()
    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    #print(label, confidence)
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img


print("Predicting images...")
webcam=cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FPS, 1)
while True:
    (_,im)=webcam.read()
    cv2.waitKey(4)
    face,rect=detect_face(im)
    #print(face,rect)
    if face is not None:
        draw_rectangle(im,rect)
        cv2.imshow('vf',cv2.resize(im, (960, 540)) )
        

'''
#load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")
test_img4 = cv2.imread("test-data/test4.jpg")
#perform a prediction
print(test_img1.shape,test_img2.shape,test_img3.shape)
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)

print("Prediction complete")

cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.destroyAllWindows()

cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(2000)
cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))
cv2.waitKey(2005)
cv2.imshow(subjects[4], cv2.resize(predicted_img4, (400, 500)))
cv2.waitKey(2005)

#cv2.destroyAllWindows()
'''




