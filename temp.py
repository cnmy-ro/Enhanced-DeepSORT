import cv2
import time

frame = cv2.imread('1.jpg')
cv2.imshow('frame', frame)
time.sleep(3)
cv2.destroyAllWindows()