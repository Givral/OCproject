import cv2
import numpy as np
from math import atan, pi
import pandas
from openpyxl import load_workbook
import array
import os
os.chdir('E:/pictures/bash')
#this version include ycrcb frame, Gaussian Blur &amp; morphologyEx
cap = cv2.VideoCapture("top0.mp4")
subtractorKNN = cv2.createBackgroundSubtractorKNN()
book = load_workbook('datatop0.xlsx')
while True:
    for i in range(2, 1000, 1):
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        # frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.
        # gray_frame = cv2.resize(gray_frame, (0, 0), fx=0.8, fy=0.

        mask4 = subtractorKNN.apply(gray_frame)

        _, mask4 = cv2.threshold(mask4, 200, 255, cv2.THRESH_BINARY)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        opening_m4 = cv2.morphologyEx(mask4, cv2.MORPH_OPEN, kernel1, iterations=1)
        closing_m4 = cv2.morphologyEx(opening_m4, cv2.MORPH_CLOSE, kernel2, iterations=3)

        contours, _ = cv2.findContours(closing_m4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c in contours:
                area = cv2.contourArea(c)
                if area > 300:
                    cv2.drawContours(frame, c, -1, (0, 255, 0), 2)
                    # draw rotated rect and calculate area of rotated rect
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(frame, [box], 0, (0, 0, 255), 1)
                    # find the length and width of rotated rect
                    (x, y), (width, height), rect_angle = rect
                    ratio = height/width if(height > width) else width/height
                    area_rect = width * height
                    ratio_area = area / area_rect
                    print("ratio of w & h:", ratio)
                    print("area:", area)
                    print("area_rect", area_rect)
                    print("ratio_area:", ratio_area)
                    angle = 90 + rect_angle if (width < height) else -rect_angle
                    print("angle of rect:", angle)
                    # fitting a line and calculate the angle of line
                    rows, cols = frame.shape[:2]
                    [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((cols - x) * vy / vx) + y)
                    cv2.line(frame, (cols - 1, righty), (0, lefty), (0, 0, 255))
                    angle_line = atan((abs(righty - lefty)) / (abs(cols - 1)))*180/pi

                    print("angle of line", angle_line)

                    data = pandas.DataFrame(
                        {'ratio_w_h': [ratio], 'area': [area], 'area_rect': [area_rect], 'ratio_area': [ratio_area],
                         'angle_of_rect': [angle],'angle_of_line': [angle_line]})
                    writer = pandas.ExcelWriter('datatop0.xlsx', engine='openpyxl')
                    writer.book = book
                    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                    data.to_excel(writer, startrow=i, header=False, index=False)
                    writer.save()

    cv2.imshow("Frame", frame)
    # cv2.imshow("Basic method", difference)
    cv2.imshow("KNN", closing_m4)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()