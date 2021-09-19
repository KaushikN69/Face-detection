import cv2
from random import randrange

#detects face (classifier)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') 


#to capture video from webcam
webcam =  cv2.VideoCapture(0)

#iterate forever over frames 
while True:

    successful_frame_read , frame = webcam.read()

    #must convert to gray scale
    grayscale_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordeinates = trained_face_data.detectMultiScale(grayscale_img)

    for (x,y,w,h) in face_coordeinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

    cv2.imshow('face detection',frame)
    key = cv2.waitKey(1)#1 here is a millisecond time take for per frame
     
    if key==81 or key ==113:
        break

webcam.release()
    

