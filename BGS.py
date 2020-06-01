import cv2
import numpy as np
import math

"""
Function to create background subtraction
"""
# function to create BGS
subtractorKNN = cv2.createBackgroundSubtractorKNN()
subtractorKNN1 = cv2.createBackgroundSubtractorKNN()
subtractorKNN2 = cv2.createBackgroundSubtractorKNN()

# function to create morphology
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def backgroundsub(frame):
    #convert from BGR to Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # apply Gaussian Blur with kernel 3x3
    gaussian_blur = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    #apply KNN Background subtraction
    mask = subtractorKNN.apply(gaussian_blur)
    #remove shadow with thresholding
    _, threshold = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #apply opening morphology
    opening_m = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel1, iterations=1)
    #apply closing morphology
    closing_m = cv2.morphologyEx(opening_m, cv2.MORPH_CLOSE, kernel2, iterations=3)
    #return background subtracted image
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
