import cv2
import numpy as np
import math
import os
os.chdir('E:/pictures/bash')
PI = 3.141592653589793
Hf = 480
Wf = 640
R = float(240)
Cfx = float(320)
Cfy = float(240)
He = 240
We = int(2*PI*R)

# function to create BGS
subtractorKNN = cv2.createBackgroundSubtractorKNN()

# function to create morphology
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
def findFisheye(R,Cfx,Cfy,He,We):

    map_x = np.zeros((He, We), np.float32)
    map_y = np.zeros((He, We), np.float32)
    for Xe in range(0, We-1):
        for Ye in range(1, He - 71):
            r = (He-Ye) / He * R
            theta = (We-Xe) / We * 2.0 * PI
            Xf = Cfx + r * math.sin(theta)
            Yf = Cfy + r * math.cos(theta)
            map_x.itemset((Ye, Xe), int(Xf))
            map_y.itemset((Ye, Xe), int(Yf))

    return map_x, map_y


def unwarp(cap, xmap, ymap):

    output = cv2.remap(cap,xmap,ymap,cv2.INTER_LINEAR)
    return output


xmap, ymap = findFisheye(R,Cfx,Cfy,He,We)
cap = cv2.VideoCapture("top0.mp4")
while True:
    for i in range(2, 1000, 1):
        _, fisheyeImage = cap.read()
        gray_frame = cv2.cvtColor(fisheyeImage, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        mask = subtractorKNN.apply(gaussian_blur)
        _, threshold = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        opening_m = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel1, iterations=1)
        closing_m = cv2.morphologyEx(opening_m, cv2.MORPH_CLOSE, kernel2, iterations=3)
        output= unwarp(closing_m, xmap, ymap)
        #output = cv2.resize(output,(int(We/2),int(He/2)))
        #fisheyeImage = cv2.resize(fisheyeImage, (int(Wf / 2), int(Hf / 2)))
        #cv2.imshow("fisheye", fisheyeImage)
        cv2.imshow("pano", output)
        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()






