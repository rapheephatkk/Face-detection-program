import cv2

cap = cv2.VideoCapture("video.mp4")
face = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eyes = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

while (cap.isOpened()) :
    ref , frame = cap.read()
    if ref == True :
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
        face_detect = face.detectMultiScale(gray , 1.1 , 5)
        eyes_detect = eyes.detectMultiScale(gray , 1.1 , 4)
        for (x,y,w,h) in face_detect :
            cv2.rectangle(frame,(x , y),(x + w , y + h) ,  (0,255,0) , 5)
            cv2.putText(frame , "Face" , (x-10 , y -10) , 2 , 0.5 , (255,255,255))
            for (ex , ey , ew , eh) in eyes_detect :
                cv2.rectangle(frame,(ex , ey),(ex + ew , ey + eh) ,  (0,0,255) , 3)
                cv2.putText(frame , "Eyes" , (ex-10 , ey -10) , 2 , 0.5 , (255,255,255))
    cv2.imshow("Output",frame)
    if cv2.waitKey(8) & 0xFF == ord("e") :
        break

cap.release()
cv2.destroyAllWindow()