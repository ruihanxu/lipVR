import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

cap = cv2.VideoCapture(0)
ds_factor = 0.5

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # use the CascadeClassifier to find the mouth
    # we can also train ourselves' if needed
    # check openCV's library
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        # draw rec from (x,y) to (x+w,y+h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        
        # find harris corner
        gray_corner = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        frame[dst>0.01*dst.max()] = [0,0,255]
        # TODO: find corner with specific angle degree
        # so that it will only find mouth_left,mouth_right, mouth_up, mouth_down
        
        break
    
    # chop the image to only show the mouth part
    # CHEAT! as it's still output a bunch of useless corners outside the mouth
    # TODO: find a real way to only detect the corners in the green rectangle
    cv2.imshow('Mouth Detector', frame[y:y+h,x:x+w])

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
