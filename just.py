import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while(1):
    ret, frame = cap.read()
    if ret == 1:

        cv2.rectangle(frame, (600, 600), (200, 200), (0, 0, 255), 0)

        box = frame[100:600, 100:600]
        gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (35, 35), 0)
        _, th = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        _, cnt, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = np.array(cnt)

        contourOutline = np.zeros(box.shape, np.uint)
        cv2.drawContours(contourOutline, cnt, -1, (0, 255, 0), 0)


        cv2.imshow('frameGrayScl', gray)
        cv2.imshow('frameGaussBlurr', blur)
        cv2.imshow('frameThreshd', th)
        if cv2.waitKey(1) & 0xFF == ord('`'):
            break
    else:
        break


cap.release()

cv2.destroyAllWindows()
