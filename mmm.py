import cv2
import numpy as np
import os
os.chdir('E:/pictures/bash')
# capture video
cap = cv2.VideoCapture("top0.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('topone70pano.mp4', fourcc, 10,(640,480))
# function to create BGS
subtractorKNN = cv2.createBackgroundSubtractorKNN()
# function to create morphology
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#writing file header

while True:
    for i in range(2, 1000, 1):
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_blur = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        mask = subtractorKNN.apply(gaussian_blur)
        _, threshold = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        opening_m = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel1, iterations=1)
        closing_m = cv2.morphologyEx(opening_m, cv2.MORPH_CLOSE, kernel2, iterations=3)
        contours, _ = cv2.findContours(closing_m, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        out.write(closing_m)
        cv2.imshow("top0",closing_m)
        key = cv2.waitKey(30)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()









