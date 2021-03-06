
import cv2
import numpy as np
import sys
sys.path.append("/home/givral/Documents/OCProject")
sys.path.append("/home/givral/Documents")
import os
path = "/home/givral/Documents/OCProject"
os.chdir(path)
os.environ['PATH'] += ':'+path
path = "/home/givral/Documents"
os.chdir(path)
os.environ['PATH'] += ':'+path
from OCProject import BGS
from OCProject import fish2pano as f2p
import pandas
from openpyxl import load_workbook
from math import atan
from OCProject import export_data


#global variables
PI = 3.141592653589793
#height la chieu cao , width la chieu ngang
#height of fisheye image
Hf = 480
#width of fisheye image
Wf = 640
#radius of fisheye image
R = float(Hf/2)
# (Cfx,Cfy) is the center of the coordinate of fisheye image
Cfx = float(Wf/2)
Cfy = float(Hf/2)
#He : Height of  equirectangular image
He = int(Hf/2)
#We : Width of equirectangular image
We = int(2*PI*R)
# radius r of inner circle
r = 70
#starting frame number
frame_no = 0
font = cv2.FONT_HERSHEY_SIMPLEX
#Path to the dataset
PATH_TO_DATASET = '/home/givral/PycharmProjects/OCproject/datasets/BOMNI/scenario1'

#load workbook
book = load_workbook('/home/givral/Documents/OCProject/data/result/top_4_r70_unwrap_first.xlsx')
export_data.write_header(book)

# Create a map of panoramic coordinate
xmap, ymap = f2p.findFisheye(R,Cfx,Cfy,He,We,r)

# Capture video
cap = cv2.VideoCapture('/home/givral/Documents/OCProject/data/bomni-5840/scenario1/top-4.mp4')
while cap.isOpened():
    # read the frame
    _, fisheyeImage = cap.read()
    # crop the center of the fish eye image with radius r
    cropped = f2p.cropcircle(fisheyeImage,int(Wf/2),int(Hf/2),70)
    # apply BGS on cropped image
    cropped = BGS.backgroundsub(cropped)
    # panoramic image with no BGS
    output_nobgs = f2p.unwarp(fisheyeImage, xmap, ymap)
    # apply BGS to panoramic image
    output_nobgs_then_bgs = BGS.backgroundsub1(output_nobgs)
    # use background subtraction on fisheye image
    BGS_fisheyeImage = BGS.backgroundsub2(fisheyeImage)
    # show original fisheye image
    #cv2.imshow("fisheye", fisheyeImage)

    contours_crop, _ = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_fish, _ = cv2.findContours(BGS_fisheyeImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_unwrap_bgs, _ = cv2.findContours(output_nobgs_then_bgs, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  
    # unwrap then BGS
    for c_unwrap_bgs in contours_unwrap_bgs:
        check = 0
        area_unwrap_bgs = cv2.contourArea(c_unwrap_bgs)
        for c_crop in contours_crop:
            area_crop = cv2.contourArea(c_crop)
            if area_crop > 30:
                check = 1
        if check == 1:
            break
        else:
            if area_unwrap_bgs > 150:
                cv2.drawContours(output_nobgs_then_bgs, contours_unwrap_bgs, -1, (255, 255, 255), 2)
                # draw rotated rect and calculate area of rotated rect
                rect = cv2.minAreaRect(c_unwrap_bgs)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(output_nobgs_then_bgs, [box], 0, (255, 255, 255), 1)
                (x, y), (width, height), rect_angle = rect
                ratio = height / width
                area_rect = width * height
                ratio_area = area_unwrap_bgs / area_rect
                print("ratio of w & h:", ratio)
                print("area:", area_unwrap_bgs)
                print("area_rect", area_rect)
                print("ratio_area:", ratio_area)
                angle = 90 - rect_angle if (width < height) else -rect_angle
                print("angle of rect:", angle)
                rows, cols = output_nobgs_then_bgs.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(c_unwrap_bgs, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                cv2.line(output_nobgs_then_bgs, (cols - 1, righty), (0, lefty), (255, 255, 255))
                angle_line = atan((abs(righty - lefty)) / (abs(cols - 1)))
                print("angle of line", angle_line)
                camera_type = 'pano'
                data = pandas.DataFrame(
                    {'ratio_w_h': [ratio], 'area': [area_unwrap_bgs], 'area_rect': [area_rect], 'ratio_area': [ratio_area],
                     'angle_of_rect': [angle], 'angle_o_line': [angle_line], 'camera_type': [camera_type]})
                writer = pandas.ExcelWriter('/home/givral/Documents/OCProject/data/result/top_4_r70_unwrap_first.xlsx', engine='openpyxl')
                writer.book = book
                writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                data.to_excel(writer, startrow=frame_no+2, header=False, index=False)
                writer.save()

    # original fisheye image
    for c_fish in contours_fish:
        area_fish = cv2.contourArea(c_fish)
        for c_crop in contours_crop:
            area_crop = cv2.contourArea(c_crop)
            if area_crop > 30:
                if area_fish > 300:
                    cv2.drawContours(BGS_fisheyeImage, c_fish, -1, (255, 255, 255), 2)
                    # draw rotated rect and calculate area of rotated rect
                    rect = cv2.minAreaRect(c_fish)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(BGS_fisheyeImage, [box], 0, (255, 255, 255), 1)
                    (x, y), (width, height), rect_angle = rect
                    ratio = height / width
                    area_rect = width * height
                    ratio_area = area_fish / area_rect
                    print("ratio of w & h:", ratio)
                    print("area:", area_fish)
                    print("area_rect", area_rect)
                    print("ratio_area:", ratio_area)
                    angle = 90 - rect_angle if (width < height) else -rect_angle
                    print("angle of rect:", angle)
                    rows, cols = fisheyeImage.shape[:2]
                    [vx, vy, x, y] = cv2.fitLine(c_fish, cv2.DIST_L2, 0, 0.01, 0.01)
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((cols - x) * vy / vx) + y)
                    cv2.line(BGS_fisheyeImage, (cols - 1, righty), (0, lefty), (255, 255, 255))
                    angle_line = atan((abs(righty - lefty)) / (abs(cols - 1)))
                    print("angle of line", angle_line)
                    camera_type1 = 'fisheye'
                    data = pandas.DataFrame(
                        {'ratio_w_h': [ratio], 'area': [area_fish], 'area_rect': [area_rect],
                         'ratio_area': [ratio_area],
                         'angle_of_rect': [angle], 'angle_o_line': [angle_line], 'camera_type': [camera_type1]})
                    writer = pandas.ExcelWriter('/home/givral/Documents/OCProject/data/result/top_4_r70_unwrap_first.xlsx', engine='openpyxl')
                    writer.book = book
                    writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
                    data.to_excel(writer, startrow=frame_no+2, header=False, index=False)
                    writer.save()
    cv2.putText(cropped,
            str(frame_no),
            (50, 50),
            font, 1,
            (255, 255, 255),
            2,
            cv2.LINE_4)
    frame_no += 1

    output_nobgs_then_bgs= cv2.resize(output_nobgs_then_bgs, (int(We / 2), int(He / 2)))
    cv2.imshow("fisheye", fisheyeImage)
    cv2.imshow("BGS_fisheye", BGS_fisheyeImage)
    cv2.imshow("unwrapfirst",output_nobgs_then_bgs)
    cv2.imshow("cropped", cropped)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()






