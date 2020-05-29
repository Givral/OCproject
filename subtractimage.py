import cv2
import os
import matplotlib
os.chdir('E:/pictures/bash')
i =0
while (i<=1000):

    img1 = cv2.imread('topzero'+str(i)+'.png',1)
    img2 = cv2.imread('topzero70cut'+str(i)+'.png',1)
    frame =cv2.subtract (img1,img2)
    cv2.imwrite('topzero70radius' + str(i) + '.png', frame)
    i += 1
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
