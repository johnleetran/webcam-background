import cv2
import numpy as np

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()
history = 100

background_capture = cv2.VideoCapture(0)

while True:

frame = get_frame(cap, 0.5)

ret, background = background_capture.read()
background = cv2.resize(background, (640, 480), interpolation=cv2.INTER_AREA)
ret, frame = cap.read()
# gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mask = fgbg.apply(frame, learningRate=1.0//history)
#convert from grayscale to RGB
gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


bitwise = cv2.bitwise_and(background, background, mask=mask)
gaussian = cv2.GaussianBlur(bitwise, (5, 5), 10)
add = cv2.add(background, gaussian)
add = cv2.resize(add, (1366, 768), interpolation=cv2.INTER_AREA)


cv2.imshow('add', add)

k = cv2.waitKey(30) & 0xff
if k == 27:
   break

cap.release()
cv2.destroyAllWindows()
