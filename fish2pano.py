import cv2
import numpy as np
import math
import os
"""
THIS IS A LIBRARY TO INCLUDE ALL UTILITIES OF THE PROJECT
"""

"""
Functions to create panoramic image from fisheye
"""
PI = 3.141592653589793
#function to find fisheye coordiante
def findFisheye(R, Cfx, Cfy, He, We,r):

    map_x = np.zeros((He, We), np.float32)
    map_y = np.zeros((He, We), np.float32)
    for Xe in range(0, We-1):
        for Ye in range(1, int(He) - 71):
            r = (He-Ye) / He * R
            theta = (We-Xe) / We * 2.0 * PI
            Xf = Cfx + r * math.sin(theta)
            Yf = Cfy + r * math.cos(theta)
            map_x.itemset((Ye, Xe), int(Xf))
            map_y.itemset((Ye, Xe), int(Yf))

    return map_x, map_y

#function to unwrap the fisheye to panoramic view
def unwarp(cap, xmap, ymap):

    output = cv2.remap(cap,xmap,ymap,cv2.INTER_LINEAR)
    return output

"""
Function to create video from images
"""

def create_video_from_img():
    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter('topone70pano.mp4', fourcc, 10, (640, 480))

    for j in range(0, 1000):
        img = cv2.imread('topone70pano' + str(j) + '.png')
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

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
