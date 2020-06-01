import numpy as np
import cv2

cap = cv2.VideoCapture('/home/givral/PycharmProjects/OCproject/datasets/BOMNI/scenario1/top-0.mp4')
frame_no = 0
while cap.isOpened():

    # Capture frames in the video
    ret, frame = cap.read()

    # describe the type of font
    # to be used.
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use putText() method for
    # inserting text on video
    cv2.putText(frame,
                str(frame_no),
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)

    # Display the resulting frame
    cv2.imshow('video', frame)
    frame_no += 1

    # creating 'q' as the quit
    # button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()
