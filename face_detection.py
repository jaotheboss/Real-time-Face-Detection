# face detection

import cv2
import os
os.chdir('/Users/jaoming/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades')

# initialise the classifier
# cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascPath)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# apply the cascade algo on webcam frames
video_capture = cv2.VideoCapture(0)

while True:
       # capture frame-by-frame
       ret, frames = video_capture.read()                             # each loop is one frame

       gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)                # creating a gray image from the webcam image

       faces = face_cascade.detectMultiScale(                         # detects the face using the gray image
              gray,
              scaleFactor = 1.1,
              minNeighbors = 5,
              minSize = (30, 30),
              flags = cv2.CASCADE_SCALE_IMAGE
       )

       # drawing the rectangle around the faces
       for (x, y, w, h) in faces:
              cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)

              # drawing the rectangle around the eyes
              ## looking within the areas of the face rectangle
              roi_gray = gray[y:y+h, x:x+h]
              roi_color = frames[y:y+h, x:x+h]
              eyes = eye_cascade.detectMultiScale(roi_gray)
              for (ex, ey, ew, eh) in eyes:
                     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
       
       # displaying the resulting frame
       cv2.imshow('Video', frames)

       if cv2.waitKey(1) and 0xFF == ord('q'):
              break

video_capture.release()
cv2.destroyAllWindows()