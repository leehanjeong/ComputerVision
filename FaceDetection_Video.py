import cv2
import numpy as np

face_cascade_name = "C:/Users/rosy0/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
eyes_cascade_name = "C:/Users/rosy0/anaconda3/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml"
file_name = "./video/nctVideo.mp4"

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 채널 여러개면 정확도 떨어지기 때문에
    frame_gray = cv2.equalizeHist(frame_gray) # histogram을 이용해서 좀 더 단순화
    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray) # frame에서 얼굴 추출
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4) # 얼굴은 초록 사각
        faceROI = frame_gray[y:y+h, x:x+w] # Reader Of Interest
        
        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4) #눈은 파란 원
            
    cv2.imshow("Capture - Face detection", frame)
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #image = Image.fromarray(image)
    #imgtk = ImageTk.PhotoImage(image=image)
    #detection.config(image=imgtk)
    #detection.image = imgtk

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

# Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

cap = cv2.VideoCapture(file_name)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    # Hit 'q' on the Keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
