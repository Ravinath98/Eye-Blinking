import cv2
import numpy as np
import dlib
from math import hypot

#define a video capture object
cap=cv2.VideoCapture(0) #web Camera
detector=dlib.get_frontal_face_detector()

predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# capture the frames..
while True:
    #ret--> it's a flag,whether frame was read correctly or not..
    #frame--> frame it self
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)# convert frames to grayscale(GRAY IMAGES ARE LIGHTER(to save computational power))

    faces = detector(gray)

    #retrieve all the faces from the image and render a rectangle over each face
    for face in faces: #caught multiple faces in the screen...
        #print(face.left(),face.top(),face.bottom(),face.right())
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()

        #mark rectangle around the face
        #cv2.rectangle(frame, (x1, y1),(x2, y2), color=(0, 0, 255), thickness=4)
        landmarks=predictor(gray,face)
        #print the position of the 36th point...
        #print(landmarks.part(36)) #landmarks variable is a object
        #x=landmarks.part(36).x
        #y=landmarks.part(36).y
        #cv2.circle(frame,(x,y),3,(0,0,255),2) # 3->radius, 2->thickness

        ####Right eye
        left_point_r=landmarks.part(36).x,landmarks.part(36).y
        right_point_r=landmarks.part(39).x,landmarks.part(39).y
        #upper_left=landmarks.part(37).x,landmarks.part(37).y
        #upper_right=landmarks.part(38).x,landmarks.part(38).y
        #bottom_left=landmarks.part(41).x,landmarks.part(41).y
        #bottom_right=landmarks.part(40).x,landmarks.part(40).y
        mid_upper_r=int((landmarks.part(37).x+landmarks.part(38).x)/2),int((landmarks.part(37).y+landmarks.part(38).y)/2)
        mid_bottom_r=int((landmarks.part(40).x+landmarks.part(41).x)/2),int((landmarks.part(40).y+landmarks.part(41).y)/2)
        hori_line_r=cv2.line(frame,left_point_r,right_point_r,(0,255,0),2)
        #print('left point',left_point,'  right point',right_point)
        #print('mid upper',mid_upper,' mid bottom',mid_bottom)
        verti_line_r=cv2.line(frame,mid_upper_r,mid_bottom_r,(0,0,255),2)

        hori_line_len_r=hypot(left_point_r[0]-right_point_r[0],left_point_r[1]-right_point_r[1])
        verti_line_len_r=hypot(mid_upper_r[0]-mid_bottom_r[0],mid_upper_r[1]-mid_bottom_r[1])
        #print('verical line length',verti_line_len)
        #print('horizontal line length',hori_line_len)
        #print(hori_line_len/verti_line_len)
        lines_ratio_r=hori_line_len_r/verti_line_len_r
        if lines_ratio_r>4:
            cv2.putText(frame,"Blinking-Right Eye",(20,80),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0))



        ###Left eye
        left_point_l = landmarks.part(42).x, landmarks.part(42).y
        right_point_l = landmarks.part(45).x, landmarks.part(45).y
        mid_upper_l = int((landmarks.part(43).x + landmarks.part(44).x) / 2), int((landmarks.part(43).y + landmarks.part(44).y) / 2)
        mid_bottom_l = int((landmarks.part(46).x + landmarks.part(47).x) / 2), int((landmarks.part(46).y + landmarks.part(47).y) / 2)
        hori_line_l  = cv2.line(frame, left_point_l , right_point_l , (0, 255, 0), 2)
        verti_line_l  = cv2.line(frame, mid_upper_l, mid_bottom_l, (0, 0, 255), 2)
        hori_line_len_l = hypot(left_point_l[0] - right_point_l[0], left_point_l[1] - right_point_l[1])
        verti_line_len_l  = hypot(mid_upper_l[0] - mid_bottom_l[0], mid_upper_l[1] - mid_bottom_l[1])

        lines_ratio_l = hori_line_len_l / verti_line_len_l
        if lines_ratio_l > 4:
            cv2.putText(frame, "Blinking-Left Eye", (20, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

        #Gaze Detection
        #right eye
        right_eye_region=np.array([ #use numpy array for less computation power..
            (landmarks.part(36).x,landmarks.part(36).y),
            (landmarks.part(37).x,landmarks.part(37).y),
            (landmarks.part(38).x,landmarks.part(38).y),
            (landmarks.part(39).x,landmarks.part(39).y),
            (landmarks.part(40).x,landmarks.part(40).y),
            (landmarks.part(41).x,landmarks.part(41).y)
        ])
        cv2.polylines(frame,[right_eye_region],True,(0,255,0),2)

        #left eye
        left_eye_region = np.array([  # use numpy array for less computation power..
            (landmarks.part(42).x, landmarks.part(42).y),
            (landmarks.part(43).x, landmarks.part(43).y),
            (landmarks.part(44).x, landmarks.part(44).y),
            (landmarks.part(45).x, landmarks.part(45).y),
            (landmarks.part(46).x, landmarks.part(46).y),
            (landmarks.part(47).x, landmarks.part(47).y)
        ])
        cv2.polylines(frame, [left_eye_region], True, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)  # Display the resulting frame
    key = cv2.waitKey(1)
    if key == 27:  # click esc key to exit
        break
cap.release()
cv2.destroyAllWindows()