import cv2
import numpy as np
from OCProject import fish2pano as f2p
from OCProject import BGS
from OCProject import puttext
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

#load workbook
book = load_workbook('/home/givral/PycharmProjects/OCproject/result/top_0_r70_BGS_first.xlsx')
export_data.write_header(book)
# Create a map of panoramic coordinate
xmap, ymap = f2p.findFisheye(R,Cfx,Cfy,He,We)
# Capture video
cap = cv2.VideoCapture('/home/givral/PycharmProjects/OCproject/datasets/BOMNI/scenario1/top-0.mp4')

while cap.isOpened():
    # read the frame
    _, fisheyeImage = cap.read()

    # crop the center of the fish eye image with radius r
    cropped = f2p.cropcircle(fisheyeImage,int(Wf/2),int(Hf/2),r)
    # apply BGS on cropped image
    cropped = BGS.backgroundsub(cropped)
    # show original fisheye image
    cv2.imshow("fisheye", fisheyeImage)
    # use background subtraction on fisheye image
    BGS_fisheyeImage = BGS.backgroundsub1(fisheyeImage)
    # unwrap on the background subtracted fisheye image
    output = f2p.unwarp(BGS_fisheyeImage, xmap, ymap)

    contours_crop, _ = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_fish, _ = cv2.findContours(BGS_fisheyeImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_bgs_unwrap, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # BGS then unwrap
    for c_bgs_unwrap in contours_bgs_unwrap:
        check = 0
        area_bgs_unwrap = cv2.contourArea(c_bgs_unwrap)
        for c_crop in contours_crop:
            area_crop = cv2.contourArea(c_crop)
            if area_crop > 30:
                check = 1
        if check == 1:
            break
        else:
            if area_bgs_unwrap > 150:
                cv2.drawContours(output, c_bgs_unwrap, -1, (255, 255, 255), 2)
                # draw rotated rect and calculate area of rotated rect
                rect = cv2.minAreaRect(c_bgs_unwrap)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(output, [box], 0, (255, 255, 255), 1)
                (x, y), (width, height), rect_angle = rect
                ratio = height / width
                area_rect = width * height
                ratio_area = area_bgs_unwrap / area_rect
                print("ratio of w & h:", ratio)
                print("area:", area_bgs_unwrap)
                print("area_rect", area_rect)
                print("ratio_area:", ratio_area)
                angle = 90 - rect_angle if (width < height) else -rect_angle
                print("angle of rect:", angle)
                rows, cols = output.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(c_bgs_unwrap, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)
                cv2.line(output, (cols - 1, righty), (0, lefty), (255, 255, 255))
                angle_line = atan((abs(righty - lefty)) / (abs(cols - 1)))
                print("angle of line", angle_line)
                camera_type = 'pano'
                data = pandas.DataFrame(
                    {'ratio_w_h': [ratio], 'area': [area_bgs_unwrap], 'area_rect': [area_rect], 'ratio_area': [ratio_area],
                     'angle_of_rect': [angle], 'angle_o_line': [angle_line], 'camera_type': [camera_type],'frame_no':[frame_no]})
                writer = pandas.ExcelWriter('/home/givral/PycharmProjects/OCproject/result/top_0_r70_BGS_first.xlsx', engine='openpyxl')
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
                    rows, cols = BGS_fisheyeImage.shape[:2]
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
                    writer = pandas.ExcelWriter('/home/givral/PycharmProjects/OCproject/result/top_0_r70_BGS_first.xlsx', engine='openpyxl')
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
    # inserting text on video

    output = cv2.resize(output,(int(We/2),int(He/2)))
    #fisheyeImage = cv2.resize(fisheyeImage, (int(Wf / 2), int(Hf / 2)))
    #output_nobgs_then_bgs= cv2.resize(output_nobgs_then_bgs, (int(We / 2), int(He / 2)))
    cv2.imshow("BGS_fisheye", BGS_fisheyeImage)
    cv2.imshow("BGSfirst", output)
    cv2.imshow("cropped", cropped)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()






