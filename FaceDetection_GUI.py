import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

# PIL은 이미지 프로세싱을 쉽게 할 수 있도록 도와주는 라이브러리

face_cascade_name = "C:/Users/rosy0/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
eyes_cascade_name = "C:/Users/rosy0/anaconda3/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml"
file_name = "./image/MarkWoo.jpg"
title_name = "Haar cascade object detection"
frame_width = 500

face_cascade = cv2.CascadeClassifier()
eyes_cascade = cv2.CascadeClassifier()

def selectFile():
    file_name = filedialog.askopenfilename(initialdir = "./image", title = "Select file", filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*"))) # 사진파일 선택하는 다이얼로그. init 경로는 ./image
    print("File name : ", file_name)
    read_image = cv2.imread(file_name)
    (height, width) = read_image.shape[:2] 
    frameSize = int(sizeSpin.get())
    ratio = frameSize /width # width에따라 height 정할 수 있게 해주는 비
    dimension = (frameSize, int(height * ratio))
    read_image = cv2.resize(read_image, dimension, interpolation = cv2.INTER_AREA) # 해상도에 따라 정확도가 달라짐
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image) # array를 image로 변환 ---> 이거 왜 필요한건지 모르겠
    imgtk = ImageTk.PhotoImage(image=image) # GUI에서 이용 가능한 photoimage로 만들어
    detectAndDisplay(read_image)
    
def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 채널 여러개면 정확도 떨어지기 때문에
    frame_gray = cv2.equalizeHist(frame_gray) # histogram을 이용해서 좀 더 단순화
    # Detect faces
    faces = face_cascade.detectMultiScale(frame_gray) # frame에서 얼굴 추출
    for (x, y, w, h) in faces: # x,y가 왼쪽 맨 위에 있는 점. w, h가 width, height
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
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk) # config는 Label 수정할 때 사용. 여기서는 이미지 바꾸는 용도.
    detection.image = imgtk

# main
main = Tk() # main GUI 생성
main.title(title_name)
main.geometry()

# 이미지 읽고 가공
read_image = cv2.imread(file_name)
(height, width) = read_image.shape[:2]
ratio = frame_width / width
dimension = (frame_width, int(height * ratio))
read_image = cv2.resize(read_image, dimension, interpolation = cv2.INTER_AREA) # 이미지 사이즈 조정

image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)

# GUI
label = Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0, column=0, columnspan=4) # grid는 위치 지정 
sizeLabel=Label(main, text="Frame Width : ")  # Label은 텍스트 혹은 이미지
sizeLabel.grid(row=1, column=0)
sizeVal = IntVar(value=frame_width)
sizeSpin = Spinbox(main, textvariable=sizeVal, from_=0, to=2000, increment=100, justify=RIGHT) # 스피너
sizeSpin.grid(row=1, column=1)
Button(main, text="File Select", height=2, command=lambda:selectFile()).grid(row=1, column=2, columnspan=2) # 여기서 selectFile() 함수 호출
detection = Label(main, image=imgtk)
detection.grid(row=2, column=0, columnspan=4)

# Load the cascades
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)

detectAndDisplay(read_image)

        
