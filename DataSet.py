import numpy as np
import cv2
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
name=input("enter name ")
id=input("enter id ")
sample=0
while(True):
    ret , img =  cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces :
        sample=sample+1
        cv2.imwrite("Dataset/ID."+str(id)+"."+str(name)+"."+str(sample)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.waitKey(50)
    cv2.imshow("frame",img)
    cv2.waitKey(100)
    if(sample>20):
        break
cap.release()
cv2.destroyAllWindows()
