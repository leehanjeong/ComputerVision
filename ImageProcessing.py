import cv2
import numpy as np
print("OpenCV Version:")
# 버전 정보 확인
print(cv2.__version__)

# 이미지 불러오기
img = cv2.imread("JungWoo Kim.jpg")
print("Width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

(height, width) = img.shape[:2]
center = (width // 2, height //2)

# 이미지 화면에 나타내기
cv2.imshow("JungWoo", img)

# 그림 위치 옮기
move = np.float32([[1, 0, 100], [0, 1, 100]]) # 아래로 100, 오른쪽으로 100. 음수면 위, 왼쪽으
moved = cv2.warpAffine(img, move, (width, height))
#cv2.imshow("Moved down: +, up: - and right: +, left - ", moved)

# 그림 회전시키기
move = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated = cv2.warpAffine(img, move, (width, height))
#cv2.imshow("Rotated degrees", rotated)

# 이미지 사이즈 변경
ratio = 200.0 / width
dimension = (200, int(height * ratio))
resized = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA) # 확대는 INTER_LINEAR가 좋음
#cv2.imshow("Resized", resized)

# 이미지 대칭시키기
flipped = cv2.flip(img, 1)
#cv2.imshow("Flipped Horizontal 1, Vertical 0, both -1 ", flipped)

# 이미지 마스킹하기
mask = np.zeros(img.shape[:2], dtype = "uint8")
cv2.circle(mask, center, 100, (255, 255, 255), -1)
#cv2.imshow("mask", mask)

masked = cv2.bitwise_and(img, img, mask = mask)
#cv2.imshow("JungWoo with mask", masked)

# 이미지 채널 조작하기
(Blue, Green, Red) = cv2.split(img) # 채널 나누기

cv2.imshow("Red Channel", Red)
cv2.imshow("Green Channel", Green)
cv2.imshow("Blue Channel", Blue)
cv2.waitKey(0)

zeros = np.zeros(img.shape[:2], dtype = "uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, Red]))
cv2.imshow("Green", cv2.merge([zeros, Green, zeros]))
cv2.imshow("Blue", cv2.merge([Blue, zeros, zeros]))
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 필터 씌우기
cv2.imshow("Gray Filter", gray)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Filter", hsv)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB Filter", lab)
cv2.waitKey(0)

BGR = cv2.merge([Blue, Green, Red]) # 채널 다시 합치기
cv2.imshow("Blue, Green and Red", BGR)

# 특정 좌표 값 불러오기
(b, g, r) = img[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

dot = img[50:100, 50:100]

# 값 변형하기 - 초록색으로
img[50:100, 50:100] = (0, 255, 0)

# 값 변형하기 - 사각형(선) 빨간색으로
cv2.rectangle(img, (150, 50), (200, 100), (0, 0, 255), 5)

# 값 변형하기 - 원 노란색으로(마지막 -1은 내부 다 색칠을 의미)
cv2.circle(img, (275, 75), 25, (0, 255, 255), -1)

# 값 변형하기 - 선 파란색으로
cv2.line(img, (100, 200), (300, 400), (255, 0, 0), 5)

cv2.putText(img, '정우야 나 마크로 갈아타기 전에 잘하자~!', (80,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)

cv2.imshow("JungWoo - draw", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
