import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

# PIL은 이미지 프로세싱을 쉽게 할 수 있도록 도와주는 라이브러리

face_cascade_name = "C:/Users/rosy0/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
eyes_cascade_name = "C:/Users/rosy0/anaconda3/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml"
file_name = "./video/nctVideo.mp4"
title_name = "Haar cascade object detection Video"
frame_width = 500
cap = cv2.VideoCapture()

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

# Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

def selectFile():
    file_name = filedialog.askopenfilename(initialdir = "./video", title = "Select file", filetypes = (("MP4 files", "*.mp4"), ("all files", "*.*")))
    print("File name : ", file_name)
    global cap
    cap = cv2.VideoCapture(file_name) # 비디오를 불러옴
    detectAndDisplay()
    
def detectAndDisplay():
    _, frame = cap.read() # 리턴(True, False), 프레임. 비디오의 한 프레임씩 읽
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 채널 여러개면 정확도 떨어지기 때문에
    frame_gray = cv2.equalizeHist(frame_gray) # histogram을 이용해서 좀 더 단순화
    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray) # frame에서 얼굴 추출
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4) # 얼굴은 초록 사각
        faceROI = frame_gray[y:y+h, x:x+w] # Region Of Interest
        
        # In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4) #눈은 파란 원
            
    #cv2.imshow("Capture - Face detection", frame)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk) # config는 Label 수정할 때 사용. 여기서는 이미지 바꾸는 용도.
    lmain.after(10, detectAndDisplay)

# main
main = Tk() # main GUI 생성
main.title(title_name)
main.geometry()

# Graphic window
label = Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0, column=0, columnspan=4)
Button(main, text="File Select", height=2, command=lambda:selectFile()).grid(row=1, column=0, columnspan=4)
imageFrame = Frame(main)
imageFrame.grid(row=2, column=0, columnspan=4)

# Capture video frames
lmain = Label(imageFrame)
lmain.grid(row=0, column=0)

main.mainloop() # starts CUI

cap = cv2.VideoCapture(file_name)
if not cap.isOpened: # 비디오 캡쳐 객체가 정상적으로 오픈 됐는지.
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
    
