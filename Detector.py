import os
import cv2
import numpy as np
from PIL import Image
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
recogn= cv2.face.LBPHFaceRecognizer_create()
recogn.read("Recognizer\\trainingData.yml")
id=0
fontFace=cv2.cv2.FONT_HERSHEY_COMPLEX_SMALL
fontScale = 3
fontColor = (0,0,255)
path = 'Dataset'

while(True):
    ret , img =cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h)in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
        id,conf = recogn.predict(gray[y:y+h,x:x+w])
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            if(id==int(os.path.split(imagePath)[-1].split('.')[1])):
                id=os.path.split(imagePath)[-1].split('.')[2]
                break
        cv2.putText(img,str(id),(x,y+h),fontFace,fontScale,fontColor)
    cv2.imshow("Face",img)
    if(cv2.waitKey(100)==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
