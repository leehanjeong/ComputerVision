#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2, random, math, time, threading
from collections import deque

Width = 640
Height = 480


prev_lines_l = [[[62, 368, 143, 313]]]
lines_l = [[[]]]
prev_lines_r = [[[560, 327, 617, 359]]]
lines_r = [[[]]]
left_grad = -1 * 0.63
right_grad = 0.63

steer_angle = 0


# draw rectangle
def draw_rectangle(img, x, y):
    cv2.rectangle(img, (x - 5, y - 5),
                  (x + 5, y + 5),
                  (0, 255, 0), 2)
    return img


def region_of_interest(img, vertices, color):
    
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, color)
    ROI_image = cv2.bitwise_and(img, mask)

    return ROI_image


def get_gradient(line):
    gradient = 0
    x1, y1, x2, y2 = line[0]

    if (x1 != x2):  # zero division
        gradient = float(y2 - y1) / float(x2 - x1)

    return gradient



def process_lines(lines, dir):
    global prev_lines_l, prev_lines_r
    real_lines = []

    if not (lines is None):
        # prev_line 설정
	    if(dir == 'left' and not (prev_lines_l is None)):
	        prev_line = prev_lines_l[0]
	    elif(dir == 'right' and not (prev_lines_r is None)):
	        prev_line = prev_lines_r[0]
	    else:
            prev_line = lines[0]

        for line in lines:
            
            x1, y1, x2, y2 = line[0]
            
            cur_gradient = get_gradient(line)
            prev_gradient = get_gradient(prev_line)

	     # 기울기 차이로 튀는 직선 제거
            if abs(abs(cur_gradient) - abs(prev_gradient)) < 0.3: 
		# 좌표 차이로 튀는 직선 제거
                if (abs(line[0][2] - prev_line[0][2]) < 35 or abs(line[0][2] - prev_line[0][2]) > 150) : 
		    # 수평선, 수직선 제거
                    if (abs(cur_gradient) > 0.1 and abs(cur_gradient) < 0.98):  
                        
                        real_lines.append(line)
                        prev_line = line
                        
                    else:
                        real_lines.append(prev_line)
                else:
                    real_lines.append(prev_line)
            else:
                real_lines.append(prev_line)

    return real_lines


 


def draw_lines(img, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 여러색으로 그림
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1), (x2, y2), color, 2)

    return img


# 평소에 사용하던 자표계와 달라서 right의 기울기가 양수, left의 기울기가 음수가 나옴.
def get_avg_gradient(lines):
    sum = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]

        sum += float(y2 - y1) / float(x2 - x1)

    avg = sum / len(lines)

    return avg


def get_avg_line(lines):
    sum = [0, 0, 0, 0]
    for line in lines:
        x1, y1, x2, y2 = line[0]

        sum[0] += x1
        sum[1] += y1
        sum[2] += x2
        sum[3] += y2

    for i in range(len(sum)):
        sum[i] /= len(lines)

    return sum


def show_gradient(img):
    left_text = 'left gradient:' + str(left_grad)
    right_text = 'right gradient:' + str(right_grad)

    cv2.putText(img, left_text, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 255), 1)
    cv2.putText(img, right_text, (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (153, 0, 255), 1)


# You are to find "left and right position" of road lanes
def process_image(frame):
    global prev_lines_l, lines_l, prev_lines_r, lines_r, left_grad, right_grad
    
    # gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)


    # left_roi
    xl_1, yl_1 = 95, 330
    xl_2, yl_2 = 240, 340
    xl_3, yl_3 = 10, 385
    xl_4, yl_4 = 125, 400

    # right_roi
    xr_1, yr_1 = 420, 340
    xr_2, yr_2 = 580, 320
    xr_3, yr_3 = 540, 400
    xr_4, yr_4 = 640, 375

    src = frame.copy()

    # gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    # cv2.calibrateCamera()
    # cv2.undistort()

    ############# 시행착오 #############
    # hsv는 효율 별로인듯
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_white = (0, 0, 0)
    # upper_white = (360, 30, 80)
    # hsv_mask = cv2.inRange(hsv, lower_white, upper_white)
    # hsv_lane = cv2.bitwise_and(frame, frame, mask=hsv_mask)
    # cv2.imshow(hsv_lane)

        # kernel = [[0.,0.,0.,0.,1.,1.,0.,0.,0.,0.],
    #           [0.,0.,0.,1.,1.,1.,1.,0.,0.,0.],
    #           [0.,0.,1.,1.,1.,1.,1.,1.,0.,0.],
    #           [0.,1.,1.,1.,1.,1.,1.,1.,1.,0.],
    #           [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
    #           [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],
    #           [0.,1.,1.,1.,1.,1.,1.,1.,1.,0.],
    #           [0.,0.,1.,1.,1.,1.,1.,1.,0.,0.],
    #           [0.,0.,0.,1.,1.,1.,1.,0.,0.,0.],
    #           [0.,0.,0.,0.,1.,1.,0.,0.,0.,0.]]
    # kernel = np.array(kernel)
    
    # erosion_img = cv2.erode(erosion_img, kernel, iterations=1)
    # cv2.imshow('erose2', erosion_img)
    # top_hat = gray - erosion_img
    # cv2.imshow('top-hat', top_hat)
    # bottom_hat = erosion_img - gray
    # cv2.imshow('bottom-hat', bottom_hat)
   
    # ersoe, dilate  
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    kernel3 = cv2.getStructuringElement((cv2.MORPH_ELLIPSE),(5,5))

    erosion_img1 = cv2.erode(gray, kernel1, iterations=1)
    dilation_img1 = cv2.dilate(erosion_img1, kernel1, iterations=1)
    lane_img1 = gray - dilation_img1
    cv2.imshow('erose 1', erosion_img1)
    cv2.imshow('dialate 1', dilation_img1)
    cv2.imshow('gray - diate 1', lane_img1)

    erosion_img2 = cv2.erode(gray, kernel2, iterations=1)
    dilation_img2 = cv2.dilate(erosion_img2, kernel2, iterations=1)   
    lane_img2 = gray - dilation_img2
    cv2.imshow('erose 2', erosion_img2)
    cv2.imshow('dialat 2', dilation_img2)
    cv2.imshow('gray - diate 2', lane_img2)

    erosion_img3 = cv2.erode(gray, kernel3, iterations=1)  
    dilation_img3 = cv2.dilate(erosion_img3, kernel3, iterations=1)
    lane_img3 = gray - dilation_img3
    cv2.imshow('erose 3', erosion_img3)
    cv2.imshow('dialate 3', dilation_img3)
    cv2.imshow('gray - diate 3', lane_img3)

    # blur 
    kernel_size = 5  
    blur = cv2.GaussianBlur(lane_img2, (kernel_size, kernel_size), 0)
    cv2.imshow('blur', blur)

    # canny
    low_threshold = 60 
    high_threshold = 80  
    canny_left = cv2.Canny(np.uint8(blur), low_threshold, high_threshold)
    canny_right = canny_left.copy()

    # roi(꼭짓점 쓰는 순서도 중요함)
    vertices_left = np.array([[(xl_1, yl_1), (xl_2, yl_2), (xl_4, yl_4), (xl_3, yl_3)]], dtype=np.int32)
    left_roi = region_of_interest(canny_left, vertices_left, 255)
    vertices_right = np.array([[(xr_1, yr_1), (xr_2, yr_2), (xr_4, yr_4), (xr_3, yr_3)]], dtype=np.int32)
    right_roi = region_of_interest(canny_right, vertices_right, 255)
    
    cv2.imshow("left_roi", left_roi)
    cv2.imshow("right_roi", right_roi)

#################################################################################################
    # HoughLinesP
    lines_l = cv2.HoughLinesP(left_roi, 1, math.pi / 180, 35, 2, 15)  # 인자 조절
    lines_r = cv2.HoughLinesP(right_roi, 1, math.pi / 180, 35, 2, 15)

    # process lines
    lines_l = process_lines(lines_l, 'left')
    lines_r = process_lines(lines_r, 'right')

    # line 검출이 안 된 경우
    if (not lines_l) and (lines_r):
        lines_l = lines_r 
    elif (not lines_r) and (lines_l):
        lines_r = lines_l
    elif (not lines_l) and (not lines_r):
        lines_l = prev_lines_l
        lines_r = prev_lines_r

    prev_lines_l = lines_l
    prev_lines_r = lines_r


    # draw hough lines
    draw_lines(src, lines_l)
    draw_lines(src, lines_r)

    # find rectangle area_1 - use pixel threshold
    # lpos, rpos = 135, 550
    # roi_mid_row = 315
    # roi_mid_col = 335
    # start_l, start_r = 290, 380 # 조절
    # ##cv2.circle(src, (start_l, 315),5, (255, 0, 0), -1)
    # ##cv2.circle(src, (start_r, 315),5, (255, 0, 0), -1)
    # end_l, end_r = xl_2, xr_1
    # area_len = 10 # 픽셀 수 확인 영역 가로,세로 길이
    # pixel_cnt_threshold = area_len * area_len * 0.2
    # prev_lpos, prev_rpos = lpos, rpos # 직사각형 놓친 경우 대비

    # for l in range(start_l, end_l, -1):
    #     area = left_roi[roi_mid_row : roi_mid_row + area_len, l : l + area_len] # 행, 열
    #     if cv2.countNonZero(area) > pixel_cnt_threshold:
    #         lpos = l

    # for r in range(start_r, end_r):
    #     area = right_roi[roi_mid_row : roi_mid_row + area_len, r : r + area_len] # 행, 열
    #     if cv2.countNonZero(area) > pixel_cnt_threshold:
    #         rpos = r
    # ##print("l:", lpos)
    # ##print("r:", rpos)

    # if abs(prev_lpos - lpos) > 100: # 중간선 잡기 방지
    #     lpos = prev_lpos
    # if abs(prev_rpos - rpos) > 100:
    #     rpos = prev_rpos

    # dst = draw_rectangle(dst, lpos, rpos, offset=Offset)

    # find rectangle area_2 - use lines
    left_line = []
    right_line = []


    # find average gradient
    left_grad = get_avg_gradient(lines_l)
    right_grad = get_avg_gradient(lines_r)

    # find average line
    left_line = get_avg_line(lines_l)
    right_line = get_avg_line(lines_r)

    # draw rectangle
    lpos = ((left_line[0] + left_line[2]) / 2, (left_line[1] + left_line[3]) / 2)
    rpos = ((right_line[0] + right_line[2]) / 2, (right_line[1] + right_line[3]) / 2)

    draw_rectangle(src, lpos[0], lpos[1])
    draw_rectangle(src, rpos[0], rpos[1])

    # draw blue line
    cv2.line(src, (left_line[0] + 200, left_line[1] + int(left_grad * 200)), (left_line[2], left_line[3]), (255, 0, 0),
             3)
    cv2.line(src, (right_line[0] - 200, right_line[1] - int(right_grad * 200)), (right_line[2], right_line[3]),
             (255, 0, 0), 3)
    # cv2.imshow('src', src)

    ######### 대표 직선이랑 lpos, rpos 구하기 위해 가변적 roi를 써야할 것 같음, 지금은 그렇게 안 하지만 대회해서는 상황별로 roi 다르게 잡으면 좋을 것 같음. ##########

    # check ROI
    cv2.circle(src, (xl_1, yl_1), 5, (0, 0, 255), -1)
    cv2.circle(src, (xl_2, yl_2), 5, (0, 0, 255), -1)
    cv2.circle(src, (xl_3, yl_3), 5, (0, 0, 255), -1)
    cv2.circle(src, (xl_4, yl_4), 5, (0, 0, 255), -1)

    cv2.circle(src, (xr_1, yr_1), 5, (0, 0, 255), -1)
    cv2.circle(src, (xr_2, yr_2), 5, (0, 0, 255), -1)
    cv2.circle(src, (xr_3, yr_3), 5, (0, 0, 255), -1)
    cv2.circle(src, (xr_4, yr_4), 5, (0, 0, 255), -1)

    #cv2.imshow("src", src)

    return (lpos, rpos), src


def get_steer_angle(img, pos):
    global left_grad, right_grad, steer_angle

    lpos = pos[0]
    rpos = pos[1]
    midpos = (lpos[0] + rpos[0]) / 2
    print(midpos)

    # 1. 기울기
    if (left_grad <= -0.65) and (right_grad >= 0.65):
        #  2. 거리로 직진 보완
        if (midpos > 320):
            steer_angle = -5
            cv2.putText(img, "turn slightly right", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        elif (midpos < 270):
            steer_angle = 5
            cv2.putText(img, "turn slightly left", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            steer_angle = 0
            cv2.putText(img, "go straight", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    elif left_grad > 0:  # left를 잃음
        if right_grad >= 0.65:
            #  2. 거리로 직진 보완
            if (midpos > 320):
                steer_angle = -5
                cv2.putText(img, "turn slightly right", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            elif (midpos < 270):
                steer_angle = 5
                cv2.putText(img, "turn slightly left", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                steer_angle = 0
                cv2.putText(img, "go straight", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            steer_angle = 4000 / 23 * right_grad * right_grad - 8200 / 23 * right_grad + 3560 / 23 - 10
            cv2.putText(img, "turn left", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    elif right_grad < 0:  # right를 잃음
        if left_grad <= -0.65:
            #  2. 거리로 직진 보완
            if (midpos > 320):
                steer_angle = -5
                cv2.putText(img, "turn slightly right", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            elif (midpos < 270):
                steer_angle = 5
                cv2.putText(img, "turn slightly left", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            else:
                steer_angle = 0
                cv2.putText(img, "go straight", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            steer_angle = -4000 / 23 * left_grad * left_grad - 8200 / 23 * left_grad - 3560 / 23
            cv2.putText(img, "turn right", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        # 2. 거리로 직진 보완
        if (midpos > 320):
            steer_angle = -5
            cv2.putText(img, "turn slightly right", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        elif (midpos < 270):
            steer_angle = 5
            cv2.putText(img, "turn slightly left", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            steer_angle = 0
            cv2.putText(img, "go straight", (250, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return steer_angle


def draw_steer(image, steer_angle):
    global Width, Height, arrow_pic

    arrow_pic = cv2.imread('steer_arrow.png', cv2.IMREAD_COLOR)

    origin_Height = arrow_pic.shape[0]
    origin_Width = arrow_pic.shape[1]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = Height / 2
    arrow_Width = (arrow_Height * 462) / 728

    matrix = cv2.getRotationMatrix2D((origin_Width / 2, steer_wheel_center), (steer_angle) * 1.5, 0.7)
    arrow_pic = cv2.warpAffine(arrow_pic, matrix, (origin_Width + 60, origin_Height))  # 어파인 변환한 >결과 영상을 생성
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)

    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = image[arrow_Height: Height, (Width / 2 - arrow_Width / 2): (Width / 2 + arrow_Width / 2)]

    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    # cv2.imshow('2', arrow_roi)
    res = cv2.add(arrow_roi, arrow_pic)
    # cv2.imshow('3', res)
    image[(Height - arrow_Height): Height, (Width / 2 - arrow_Width / 2): (Width / 2 + arrow_Width / 2)] = res

    cv2.imshow('steer', image)


# You are to publish "steer_anlge" following load lanes
if __name__ == '__main__':
    cap = cv2.VideoCapture('kmu_track.mkv')
    time.sleep(3)

    while not rospy.is_shutdown():
        time1 = time.time()
        ret, image = cap.read()
        pos, frame = process_image(image)  # pos = (lpos, rpos), frame은 가공된 frame
        # show gradient
        show_gradient(frame)
        steer_angle = get_steer_angle(frame, pos)
        draw_steer(frame, steer_angle)
        time2 = time.time()
        print("time:", time1 - time2)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break

