import cv2

# Capture frames in the video

    # describe the type of font
    # to be used.
font = cv2.FONT_HERSHEY_SIMPLEX


def insert_text(img,text):
    cv2.putText(img,
                'TEXT ON VIDEO',
                (50, 50),
                font, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)

