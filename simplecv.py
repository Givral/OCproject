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
subtractorKNN1 = cv2.createBackgroundSubtractorKNN()
subtractorKNN2 = cv2.createBackgroundSubtractorKNN()

# function to create morphology
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def cropcircle(img, x, y, r):
    # crop image as a square
    img = img[(y-r):y + r , x-r:x + r]
    # create a mask
    mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)
    # create circle mask, center, radius, fill color, size of the border
    cv2.circle(mask, (r, r), r, (255, 255, 255), -1)
    # get only the inside pixels
    fg = cv2.bitwise_or(img, img, mask=mask)

    mask = cv2.bitwise_not(mask)
    background = np.full(img.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=mask)
    final = cv2.bitwise_or(fg, bk)
    return final


def findFisheye(R, Cfx, Cfy, He, We):

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


def backgroundsub(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    mask = subtractorKNN.apply(gaussian_blur)
    _, threshold = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    opening_m = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel1, iterations=1)
    closing_m = cv2.morphologyEx(opening_m, cv2.MORPH_CLOSE, kernel2, iterations=3)
    return closing_m

def backgroundsub1(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    mask = subtractorKNN1.apply(gaussian_blur)
    _, threshold = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    opening_m = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel1, iterations=1)
    closing_m = cv2.morphologyEx(opening_m, cv2.MORPH_CLOSE, kernel2, iterations=3)
    return closing_m

def backgroundsub2(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    mask = subtractorKNN2.apply(gaussian_blur)
    _, threshold = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    opening_m = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel1, iterations=1)
    closing_m = cv2.morphologyEx(opening_m, cv2.MORPH_CLOSE, kernel2, iterations=3)
    return closing_m

xmap, ymap = findFisheye(R,Cfx,Cfy,He,We)
cap = cv2.VideoCapture("top0.mp4")

while cap.isOpened():
    _, fisheyeImage = cap.read()
    cropped = cropcircle(fisheyeImage,320,240,70)
    cropped = backgroundsub(cropped)
    output_nobgs = unwarp(fisheyeImage, xmap, ymap)
    output_nobgs_then_bgs = backgroundsub2(output_nobgs)
    cv2.imshow("fisheyefirst", fisheyeImage)
    fisheyeImage = backgroundsub1(fisheyeImage)
    output = unwarp(fisheyeImage, xmap, ymap)

    contours, _ = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours1, _ = cv2.findContours(fisheyeImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours3, _ = cv2.findContours(output_nobgs_then_bgs, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for c2 in contours2:
        check = 0
        area2 = cv2.contourArea(c2)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 30:
                check = 1
        if check == 1:
            break
        else:
            if area2 > 150:
                cv2.drawContours(output, c2, -1, (255, 255, 255), 2)
                # draw rotated rect and calculate area of rotated rect
                rect = cv2.minAreaRect(c2)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(output, [box], 0, (255, 255, 255), 1)

    for c3 in contours3:
        check = 0
        area3 = cv2.contourArea(c3)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 30:
                check = 1
        if check == 1:
            break
        else:
            if area3 > 150:
                cv2.drawContours(output_nobgs_then_bgs, c3, -1, (255, 255, 255), 2)
                # draw rotated rect and calculate area of rotated rect
                rect = cv2.minAreaRect(c3)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(output_nobgs_then_bgs, [box], 0, (255, 255, 255), 1)
    for c1 in contours1:
        area1 = cv2.contourArea(c1)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 30:
                if area1 > 300:
                    cv2.drawContours(fisheyeImage, c1, -1, (255, 255, 255), 2)
                    # draw rotated rect and calculate area of rotated rect
                    rect = cv2.minAreaRect(c1)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(fisheyeImage, [box], 0, (255, 255, 255), 1)


    output = cv2.resize(output,(int(We/2),int(He/2)))
    fisheyeImage = cv2.resize(fisheyeImage, (int(Wf / 2), int(Hf / 2)))
    output_nobgs_then_bgs= cv2.resize(output_nobgs_then_bgs, (int(We / 2), int(He / 2)))
    cv2.imshow("fisheye", fisheyeImage)
    cv2.imshow("BGSfirst", output)
    cv2.imshow("unwrapfirst",output_nobgs_then_bgs)
    cv2.imshow("cropped", cropped)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()






