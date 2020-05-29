import cv2
import numpy as np
import os
os.chdir('E:/pictures/bash')

img=[]
import cv2
import numpy as np

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'H264')
video = cv2.VideoWriter('topone70pano.mp4', fourcc, 10,(640,480))

for j in range(0, 1000):
    img = cv2.imread('topone70pano'+str(j) + '.png')
    video.write(img)

cv2.destroyAllWindows()
video.release()