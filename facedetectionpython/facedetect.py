import cv2
from random import randrange

#detects face (classifier)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') 


# choose an image to detect faces imread(imageread)
img = cv2.imread('djkh.jpeg')


#must convert to grayscale
grayscale_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

#2 at the end is fitness of the rectangle,(0,255,0)is green color of rectangle,(91,91..)are coordinates of the face in the pic 
#cv2.rectangle(img,(91,91),(91+211,91+211),(0, 255 , 0),2)
#for (x,y,w,h) in face_coordinates:
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#for differente coloured square boxes first give "from random import randrange"on top
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)

#print(face_coordinates)
cv2.imshow(' face detect',img)
cv2.waitKey()#waits until the image is opened


print("code complete")