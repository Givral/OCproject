import cv2
import numpy as np
from math import atan, pi
import pandas
from openpyxl import load_workbook
import array
import os
os.chdir("E:/pictures/bash")

cap = cv2.VideoCapture("top0.mp4")#full size fisheye
cap1 = cv2.VideoCapture("top0pano.mp4")#pano fisheye
cap2 = cv2.VideoCapture("top0cut.mp4")#fisheye cut
cap3 = cv2.VideoCapture("top0radius70.mp4")#fisheye cut radius 70
#book = load_workbook('datatop0.xlsx')
subtractorKNN = cv2.createBackgroundSubtractorKNN()
subtractorKNN1 = cv2.createBackgroundSubtractorKNN()
subtractorKNN2 = cv2.createBackgroundSubtractorKNN()
subtractorKNN3 = cv2.createBackgroundSubtractorKNN()
while True:
    for i in range(2,1000,1):
        _, frame = cap.read()
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        _, frame3 = cap3.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        gray_frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        gray_frame1 = cv2.GaussianBlur(gray_frame1, (3, 3), 0)
        gray_frame2 = cv2.GaussianBlur(gray_frame2, (3, 3), 0)
        gray_frame3 = cv2.GaussianBlur(gray_frame3, (3, 3), 0)

        mask = subtractorKNN.apply(gray_frame)
        mask1 = subtractorKNN1.apply(gray_frame1)
        mask2 = subtractorKNN2.apply(gray_frame2)
        mask3 = subtractorKNN3.apply(gray_frame3)

        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        _, mask1 = cv2.threshold(mask1, 200, 255, cv2.THRESH_BINARY)
        _, mask2 = cv2.threshold(mask2, 200, 255, cv2.THRESH_BINARY)
        _, mask3 = cv2.threshold(mask3, 200, 255, cv2.THRESH_BINARY)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))









