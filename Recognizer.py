import os
import numpy as np
import cv2
from PIL import Image
recognizer=cv2.face.LBPHFaceRecognizer_create()
path = 'Dataset'
def returnImage(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        img=Image.open(imagePath).convert('L')
        faceArray=np.array(img,'uint8')
        id=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceArray)
        IDs.append(id)
        cv2.imshow("training",faceArray)
        cv2.waitKey(10)
    return np.array(IDs),faces
ids,faces=returnImage(path)
print(ids)
recognizer.train(faces,ids)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
