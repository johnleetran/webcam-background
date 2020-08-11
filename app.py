import cv2
import numpy as np

currentFrame = None
base_red = 156
base_green = 135
base_blue = 108

def onMouse(event, x, y, flags, param):
    global base_red, base_green, base_blue, currentFrame
    if event == cv2.EVENT_LBUTTONUP:
        print("event: " + str(event))
        print("x: " + str(x) + " y: " + str(y))
        print("flags: " + str(flags))
        print("param: " + str(param))
        print("")
        base_red = currentFrame[y,x,2]
        base_green = currentFrame[y,x,1]
        base_blue = currentFrame[y,x,0]

def display_web_camera(window_name):
    global base_red, base_green, base_blue, currentFrame
    cameraCapture = cv2.VideoCapture(0)
    background_image = cv2.imread("image.jpg", -1)
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setMouseCallback(window_name, onMouse)
    height, width, channels = background_image.shape
    alpha = 0.5

    success, frame = cameraCapture.read()


    background_image = cv2.resize(
        background_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
    base_red = 0
    base_green = 0
    base_blue = 0
    while cv2.waitKey(1) == -1:
        if frame is not None:
            base_upper = 70
            base_lower = 50

            currentFrame = frame

            frame_copy = np.copy(frame)
            # [R value, G value, B value]
            lower_color_bound = np.array(
                [base_red-base_upper, base_green-base_upper, base_blue-base_upper])
            upper_color_bound = np.array(
                [base_red+base_upper, base_green+base_upper, base_blue+base_upper])

            mask = cv2.inRange(
                frame_copy, lower_color_bound, upper_color_bound)
            masked_image = np.copy(frame_copy)
            masked_image[mask != 0] = [0, 0, 0]

            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
            background_copy = np.copy(background_image)
            background_copy[mask == 0] = [0, 0, 0]

            frame = background_copy + masked_image

            cv2.imshow(window_name, frame)
        success, frame = cameraCapture.read()

    cv2.destroyWindow(window_name)


window_name = 'Web Cam'
display_web_camera(window_name)

