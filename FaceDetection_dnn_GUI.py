import cv2
import numpy as np
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog

model_name = 'sample/res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'sample/deploy.prototxt.txt'
min_confidence = 0.5
file_name = 'image/superM.jpg'
title_name = 'dnn Deep Learning object detection'
frame_width = 300
frame_height = 300

def selectFile():
    file_name = filedialog.askopenfilename(initialdir = "./image", title = "Select file", filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*"))) # 사진파일 선택하는 다이얼로그. init 경로는 ./image
    print("File name : ", file_name)
    read_image = cv2.imread(file_name)
    (height, width) = read_image.shape[:2] 
    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image) # array를 image로 변환 ---> 이거 왜 필요한건지 모르겠
    imgtk = ImageTk.PhotoImage(image=image) # GUI에서 이용 가능한 photoimage로 만들어
    detectAndDisplay(read_image, width, height)
    
def detectAndDisplay(frame, w, h):
    # pass the blob through the model and obtain the detections
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    # Resizing to a fixed 300x300 pixels and then normalizing it
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)) # 블롭 객체 생성
    model.setInput(blob)
    detections = model.forward()
    min_confidence = float(sizeSpin.get())
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence(i.e., proabability 확률) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the 'confidence' is greater than the minimum confidence
        if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(confidence, startX, startY, endX, endY)

            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # show the output image
    # cv2.imshow("Face Detection by dnn", frame)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image)
    detection.config(image=imgtk)
    detection.image = imgtk

# main
main = Tk()
main.title(title_name)
main.geometry()

read_image = cv2.imread(file_name)
image = cv2.cvtColor(read_image, cv2.COLOR_BGR2RGB) # openCV는 RGB를 사
image = Image.fromarray(image)
imgtk = ImageTk.PhotoImage(image=image)
(height, width) = read_image.shape[:2]

# GUI
label = Label(main, text=title_name)
label.config(font=("Courier", 18))
label.grid(row=0, column=0, columnspan=4) # grid는 위치 지정 
sizeLabel=Label(main, text="Min Confidence : ")  # Label은 텍스트 혹은 이미지
sizeLabel.grid(row=1, column=0)
sizeVal = IntVar(value=min_confidence)
sizeSpin = Spinbox(main, textvariable=sizeVal, from_=0, to=1, increment=0.05, justify=RIGHT) # 스피너
sizeSpin.grid(row=1, column=1)
Button(main, text="File Select", height=2, command=lambda:selectFile()).grid(row=1, column=2, columnspan=2) # 여기서 selectFile() 함수 호출
detection = Label(main, image=imgtk)
detection.grid(row=2, column=0, columnspan=4)
detectAndDisplay(read_image, width, height) # width, height는 왜 보내주는거

main.mainloop()


