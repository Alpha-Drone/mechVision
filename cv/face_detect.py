'''
 Title: AP Face Detection
 Files: cv/cascades

 Contributors:
    Devin Mozee
    Ronald Simpson
    Geromy Edwards

The following program dectects faces via video source using OpenCV cascades.
OpenCV must be installed in order for the program to work; see http://opencv.org/releases.html
for source pack.

Cascades are available in our repo: https://github.com/Alpha-Drone/mechVision
'''

import numpy as np
import cv2
import argparse

#create argument for vision command
ap = ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=False, help="video source")
args = vars(ap.parse_args())

#load in cascades
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")



if args['video']:

    #Capture video source
    vid = cv2.VideoCapture(int(args["video"])) #change input base on computer

    while True:

        ret, frame = vid.read()

        if ret: #let our camera warm up

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #for every face draw a square
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh), (0,255,0),2)
            cv2.imshow('video',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    vid.release()
    cv2.destroyAllWindows()
