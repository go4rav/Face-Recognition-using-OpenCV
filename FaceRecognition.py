import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eyeCascade =  cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
    ret , img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img , (x,y), (x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h , x:x+w] # converts image into grayscale
        roi_color = img[y : y+h , x : x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        for (ex , ey , ew , eh) in eyes :
            cv2.rectangle(roi_color,(ex,ey),(ex + ew , ey + eh), (0,255,0),2)
            
    cv2.imshow('image', img)
    cv2.waitKey(30) 
cap.release()
cv2.destroyAllWindows()
