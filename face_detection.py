# face detection

import cv2
import os
os.chdir('/Users/jaoming/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades') # setting up the directory that holds the .xml files (the weights for the ml model)

# initialise the classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # importing the trained weights for face detecting model
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')                   # importing the trained weights for the eye detecting model

# apply the cascade algo on webcam frames
video_capture = cv2.VideoCapture(0)                                          # accesses the webcam

while True:
       # capture frame-by-frame
       ret, frames = video_capture.read()                             # each loop is one frame. reading the image frames from the webcam

       gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)                # creating a gray image from the webcam image

       faces = face_cascade.detectMultiScale(                         # detects the face using the gray image
              gray,
              scaleFactor = 1.1,
              minNeighbors = 5,
              minSize = (30, 30),
              flags = cv2.CASCADE_SCALE_IMAGE
       )

       # drawing the rectangle around the faces
       for (x, y, w, h) in faces:                                     # retrieves the coordinate for where the face is predicted to be
              cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2) # cv2.rectangle(which image to draw on, start point, end point, color (BGR), thickness of the rectangle)

              # drawing the rectangle around the eyes
              ## looking within the areas of the face rectangle
              roi_gray = gray[y:y+h, x:x+h]                           # gray image of the face in the rectangle
              roi_color = frames[y:y+h, x:x+h]                        # colored image of the face in the rectangle
              eyes = eye_cascade.detectMultiScale(roi_gray)           # using the gray image to detect the eyes
              for (ex, ey, ew, eh) in eyes:                           # retrieves the coordinate for where the eyes are predicted to be 
                     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2) # same as above
       
       # displaying the resulting frame
       cv2.imshow('Video', frames)                                    # what image to show. since we did everything to `frames`, we show that 

       if cv2.waitKey(1) and 0xFF == ord('q'):                        # press q to quit the video. if not use ctrl+c
              break

video_capture.release()                                               # closes video capture device
cv2.destroyAllWindows()                                               # closes all the windows created by this package
