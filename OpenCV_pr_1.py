import numpy as np
import cv2

def showVideo():
    try:
        print("카메라 구동합니다.")
        cap = cv2.VideoCapture(0)
    except:
        print("카메라 구동 실패")
        return

    cap.set(3,480)
    cap.set(4,320)

    while True:
        ret, frame = cap.read() # 저장 작동중이면 ret값이 True, 아니면 False

        if not ret:
            print('비디오 읽기 오류')
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('video', gray)

        k = cv2.waitKey(1)
        if k ==27:
            break

    cap.release()
    cv2.destroyAllWindows()
showVideo()

# ========== 비디오 저장하기 또는 녹화하기 ===========
import numpy as np
import cv2

def writeVideo2():
    try:
        print('카메라 작동')
        cap = cv2.VideoCapture(1)

    except:
        print('카메라 작동 실패')
        return

    fps = 20.0 # 초당 20.0 프레임
    width = int(cap.get(3))
    height = int(cap.get(4))
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X') # 코덱 적용

    out = cv2.VideoWriter('mycam.avi', fcc, fps, (width, height))
    print('녹화를 시작합니다.')

    while True:
        ret, frame = cap.read()

        if not ret:
            print("비디오 읽기 오류")
            break

        cv2.imshow('video', frame)
        out.write(frame)

        k = cv2.waitKey(1)
        if k == 27:
            print('녹화를 종료합니다.')
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

writeVideo2()


#====================== 도형 그리기 =========================
import cv2
import numpy as np

def drawing():
    img = np.zeros((512,512,3), np.uint8) # 검정색 판 생성
    # 도형 그리기 위한 공간 생성, 데이터 타입 : uint8

    # 도형 그리기
    cv2.line(img, (0,0), (511,511), (255,0,0), 5)
    # 좌표 (0,0)에서 (511,511)까지 파란색의 두께 5인 직선

    cv2.rectangle(img, (384,0), (512,128), (0,255,0),3)
    # (384,0)의 좌측 상단 꼭지점, (510,128)의 우측하단, 색상, 두께 3

    cv2.circle(img, (447,63),63,(0,0,255), -1)
    # (447,63):원의 중심, 63:반지름, (0,0,255): 색상, -1 : 주어진 생삭으로 도형을 채움

    cv2.ellipse(img, (256,256),(100,50), 0,0,180,(255,0,0),-1) # 타원
    # (256,256): 타원의 중심, (100,50): 각각 장축과 단축의 1/2길이
    # 0,0,180 : 타원의 기울기 각도, 타원 호를 그리는 시작 각도, 타원 호를 그리는 끝 각도
    # -1 : 색상 채움

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV', (10,500), font, 4, (255,255,255), 2)
    # (공간, 위치, 폰트, 폰트 크기, 색상, 글자 굵기기)
    cv2.imshow('drawing', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

drawing()


# ============ 마우스로 도형 그리기 ==============
# ---------------------------------------------
import numpy as np
import cv2
from random import shuffle

b = [i for i in range(256)]
g = [i for i in range(256)]
r = [i for i in range(256)]

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        shuffle(b), shuffle(g), shuffle(r)
        cv2.circle(param, (x, y), 50, (b[0], g[0], r[0]),-1)

def mouseBrush():
    img = np.zeros((512,512,3), np.uint8)
    cv2.namedWindow('paint')
    cv2.setMouseCallback('paint', onMouse, param=img)

    while True:
        cv2.imshow('paint', img)
        k = cv2.waitKey(1)
        if k ==27:
            break
    cv2.destroyAllWindows()

mouseBrush()

#  업그레이드 된 그림 그리기

import numpy as np
import cv2
from random import shuffle
import math

mode, drawing = True, False

ix, iy = -1, -1 # 마우스 누른 위치
B = [i for i in range(256)]
G = [i for i in range(256)]
R = [i for i in range(256)]

def onMouse(event, x, y, flags, param):
    global ix, iy, drawing, mode, B, G, R

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        shuffle(B), shuffle(G), shuffle(R)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(param, (ix, iy), (x, y), (B[0], G[0], R[0],-1))

            else:
                r = (ix-x)**2 + (iy-y)**2
                r = int(math.sqrt(r))
                cv2.circle(param, (ix, iy), r, (B[0], G[0], R[0]), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(param, (ix, iy), (x, y), (B[0], G[0], R[0]),-1)
        else:
            r = (ix - x) ** 2 + (iy - y) ** 2
            r = int(math.sqrt(r))
            cv2.circle(param, (ix, iy), r, (B[0], G[0], R[0]), -1)

def mouseBrush():
    global mode

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('paint')
    cv2.setMouseCallback('paint', onMouse, param=img)

    while True:
        cv2.imshow('paint', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('m'):
            mode = not mode
    cv2.destroyAllWindows()

mouseBrush()


# ==================== 트랙바 활용하기 =====================
# --------------------------------------------------------

# cv2.createTrackbar(trackbarName, windowName, start, end, onChange) : 트랙바를 지정된 윈도우에 생성하는 함수
# onChange :  트릭바 이벤트 발생시 수생되는 콜백 함수

# cv2.getTrackbarPos(trackbarname, windowname) : 트랙바의 현재 위치를 리턴하는 함수

import numpy as np
import cv2

def onChange():
    pass

def trackbar():
    img = np.zeros((200, 512,3), np.uint8)
    cv2.namedWindow('Color_palette')

    cv2.createTrackbar('B', 'Color_palette', 0, 255, onChange)
    cv2.createTrackbar('G', 'Color_palette', 0, 255, onChange)
    cv2.createTrackbar('R', 'Color_palette', 0, 255, onChange)

    switch = '0: OFF\n1:ON'
    cv2.createTrackbar(switch, 'Color_palette', 0, 1, onChange)

    while True:
        cv2.imshow('Color_palette', img)
        k = cv2.waitKey(1)

        if k == 27:
            break
        b = cv2.getTrackbarPos('B', 'Color_palette')
        g = cv2.getTrackbarPos('G', 'Color_palette')
        r = cv2.getTrackbarPos('R', 'Color_palette')
        s = cv2.getTrackbarPos(switch, 'Color_palette')

        if s == 0: # 스위치가 off일때 모든 트랙바의 값을 0으로 함
            img[:] = 0
        else:
            img[:] = [b, g, r] # on일 경우
    cv2.destroyAllWindows()

trackbar()


# =========== 이미지 픽셀 조작 및 ROI 조작 =============
# ---------------------------------------------------
import numpy as np
import cv2

img = cv2.imread('images/Minions.jpg')
px = img[340,200] # 이미지의 (340,200) 위치에 색상
print(px)
# --------------------------------------------------
import numpy as np
import cv2
img = cv2.imread('images/Minions.jpg')

B = img.item(340,200,0)
G = img.item(340,200,1)
R = img.item(340,200,2)

BGR = [B, G, R]
print(BGR)

# 이미지 속성 얻기
print(img.shape)
print(img.size)
print(img.dtype)

# 이미지 ROI 설정
import numpy as np
import cv2

img = cv2.imread('images/Minions.jpg')
cv2.imshow('original', img)

subimg = img[300:400, 350:540]
cv2.imshow('cutting', subimg)

img[300:400, 0:400] = subimg

print(img.shape)
print(subimg.shape)

cv2.imshow('modified', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 채널 분할 및 합치기
import numpy as np
import cv2

img = cv2.imread('images/Minions.jpg')
b, g, r = cv2.split(img)
print(img[100, 100])
print(b[100, 100], g[100, 100], r[100,100])

# ---------------- 이미지 연산 처리를 이용한 이미지 합성하기 ------------------
import numpy as np
import cv2

img = cv2.imread('images/Minions.jpg')
b, g, r = cv2.split(img)

# 채널별로 보기
cv2.imshow('blue channel', b)
cv2.imshow('green channel', g)
cv2.imshow('red channel', r)

cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import cv2

def addImage(imgfile1, imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    add_img1 = img1 + img2
    add_img2 = cv2.add(img1, img2)

    cv2.imshow('img1+img2', add_img1)
    cv2.imshow('add(img1+img2', add_img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

addImage('images/Mini.jpg','images/tt.jpg')

# 이미지 블렌딩(Image Blending) : 가중치를 두어 합치는 방법
import numpy as np
import cv2

def onMouse(x):
    pass

def imgBlending(imgfile1, imgfile2):
    img1 = cv2.imread(imgfile1)
    img2 = cv2.imread(imgfile2)

    cv2.namedWindow('ImgPane')
    cv2.createTrackbar('MIXING', 'ImgPane', 0, 100, onMouse)
    mix = cv2.getTrackbarPos('MIXING', 'ImgPane')

    while True:
        img = cv2.addWeighted(img1, float(100-mix)/100, img2, float(mix)/100, 0)
        cv2.imshow('ImgPane', img)

        k = cv2.waitKey(1)
        if k ==27:
            break
        mix = cv2.getTrackbarPos('MIXING', 'ImgPane')
    cv2.destroyAllWindows()

imgBlending('images/Mini.jpg','images/tt.jpg')
# ------------------------------------------------------------------
# 이미지 비트 연산
import numpy as np
import cv2

def bitOperation(hpos,vpos):
    img1 = cv2.imread('images/tt.jpg')
    img2 = cv2.imread('images/starbucks.jpg')

    # 로고를 상단 위에 놓기 위해 영역 지정
    rows, cols, channels = img2.shape
    roi = img1[vpos:rows+vpos, hpos:cols+hpos]

    # 로고를 위한 마스크와 역마스크 생성하기
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # ROI에서 로고에 해당하는 부분만 검정으로 만들기
    img1_fg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # 로고 이미지에서 로고 부분만 추출 하기
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # 로고 이미지 배경을 cv.add로 투명으로 만들고 ROI에 로고 이미지 넣기
    dst = cv2.add(img1_fg, img2_fg)
    img1[vpos:rows+vpos, hpos:cols+hpos] = dst

    cv2.imshow('result', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

bitOperation(10,10)

# ----------------- 색 공간 바꾸기 및 색 추적 ----------------------------------
# 색 공간 변경하기
import numpy as np
import cv2

def hsv():
    blue = np.uint8([[[255, 0, 0]]])
    green = np.uint8([[[0, 255, 0]]])
    red = np.uint8([[[0, 0, 255]]])

    hsv_blue = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

    print("HSV for BLUE:", hsv_blue)
    print("HSV for GREEN:", hsv_green)
    print("HSV for RED:", hsv_red)


hsv()
# --------------------- 웹캠을 으로 blue, green, Red 추적 --------------------
import numpy as np
import cv2

def tracking():
    try:
        print('카메라를 구동합니다.')
        cap = cv2.VideoCapture(0)

    except:
        print('카메라 구동 실패')
        return

    while True:
        ret, frame = cap.read()

        # BGR을 HSV모드로 전환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV에서 BGR로 가정할 범위를 정의함
        lower_blue = np.array([110, 100, 100])
        upper_blue = np.array([130, 255, 255])

        lower_green = np.array([50, 100, 100])
        upper_green = np.array([10, 255, 255])

        lower_red = np.array([-10, 100, 100])
        upper_red = np.array([10, 255, 255])

        # HSV 이미지에서 청색만 or 초록색 or 빨간색만 추출하기 위한 임계값
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_red = cv2.inRange(hsv, lower_red, upper_red)

        # mask와 원본 이미지를 비트 연산함
        res1 = cv2.bitwise_and(frame, frame, mask=mask_blue)
        res2 = cv2.bitwise_and(frame, frame, mask=mask_green)
        res3 = cv2.bitwise_and(frame, frame, mask=mask_red)

        cv2.imshow('original', frame)
        cv2.imshow('BLUE', res1)
        cv2.imshow('GREEN', res2)
        cv2.imshow('RED', res3)

        k = cv2.waitKey(1)
        if k == 27:
            break
    cv2.destroyAllWindows()
tracking()