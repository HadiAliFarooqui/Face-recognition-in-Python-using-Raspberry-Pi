import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0); #not working try other no
rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer/trainningData.yml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.load('recognizer/trainnigData.yml')


id=0;
name=""
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,4,1,0,3)
while(True):
    ret,img=cam.read();#cam.read returns status var and captured image
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #to convert the color img to grey scale img
    faces=faceDetect.detectMultiScale(gray,1.3,5); #returns the cordinates of image/frame/faces
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)#detection of multiple faces and showing it in a rectangle
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            name="saad"
        elif(id==2):
            name="omer"
        elif(id==3):
            name="obama"
        else:
            name="unknown"
        cv2.cv.PutText(cv2.cv.fromarray(img),str(name),(x,y+h),font,255);#here we have to give the color of text and font of the text
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()

