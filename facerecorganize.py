import cv2 # OpenCV for image processing
# Keras for loading the pre-trained model
import cv2.data
from keras.models import model_from_json
import numpy as np 

#from keras.preprocessing import image
json_file = open('emotion_detector.json', 'r')
model_json= json_file.read()
model=model_from_json(model_json)

model.load_weights('emotion_detector2.h5')
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(img):
    feature=np.array(img)
    feature=feature.reshape(1, 48, 48, 1)
    return feature/255.0

webcam = cv2.VideoCapture(0)
labels ={ 0: 'Angry',1: 'Disguste',2: 'Fear',3: 'Cool',4: 'Neutral',5: 'Sad',6: 'Surprise'}
while True:
    i,im=webcam.read()
    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    try:
        for(p,q,r,s) in faces:
            images=grey[q:q+s, p:p+r]
            cv2.rectangle(im, (p,q), (p+r, q+s), (0, 255, 0), 2)
            image=cv2.resize(images, (48, 48))
            img=extract_features(image)
            pred=model.predict(img)
            prediction_label = labels[np.argmax(pred)]
            cv2.putText(im, '%s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255))
        cv2.imshow("Output", im)
        cv2.waitKey(27)
    except cv2.error:
        pass