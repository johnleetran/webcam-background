import cv2
import numpy as np
import blend_modes

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True


def meanshift(cap):
    cap = cv2.VideoCapture(0)

    # Capture several frames to allow the camera's autoexposure to adjust.
    for i in range(10):
        success, frame = cap.read()
    if not success:
        exit(1)

    # Define an initial tracking window in the center of the frame.
    frame_h, frame_w = frame.shape[:2]
    w = frame_w//8
    h = frame_h//8
    x = frame_w//2 - w//2
    y = frame_h//2 - h//2
    track_window = (x, y, w, h)

    # Calculate the normalized HSV histogram of the initial window.
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = None
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Define the termination criteria:
    # 10 iterations or convergence within 1-pixel radius.
    term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
    return roi_hist, track_window


def threshold_slow(T, mean, image):
    # grab the image dimensions
    h = mean.shape[0]
    w = mean.shape[1]

    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            ave = np.average(mean[y, x])
            for c in range(image.shape[2]):
                # image[y, x, c] = np.clip(alpha*image[y, x, c] + beta, 0, 255)
                if ave <= T:
                    image[y, x, c] = 0

    # return the thresholded image
    return image

def display_web_camera(window_name):
    cameraCapture = cv2.VideoCapture(0)
    backgroundImage = cv2.imread("image.jpg", -1)
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)
    cv2.setMouseCallback(window_name, onMouse)
    height, width, channels = backgroundImage.shape
    alpha = 0.5

    success, frame = cameraCapture.read()

    backgroundImage = cv2.resize(
        backgroundImage, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

    while cv2.waitKey(1) == -1 and not clicked:
        if frame is not None:
            # # Perform back-projection of the HSV histogram onto the frame.
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)


            # # Perform tracking with MeanShift.
            # num_iters, track_window = cv2.meanShift(
            #     back_proj, track_window, term_crit)
            mask = np.zeros(frame.shape, np.uint8)
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(grayFrame, 100, 255, 0)
            contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            color = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)
            grayFrame = cv2.drawContours(color, contours, -1, (0, 0, 0), cv2.FILLED)
            dst = grayFrame
            #dst = cv2.bitwise_xor(grayFrame, backgroundImage)

            # previousFrame = grayFrame
            # dst = cv2.bitwise_not(frame)
            dst = cv2.bitwise_not(grayFrame)
            dst = cv2.bitwise_xor(dst, backgroundImage)
            #dst = cv2.bitwise_or(grayFrame, frame)
            alpha = 0.75
            beta = (1.0 - alpha)
            #dst = cv2.addWeighted(dst, alpha, backgroundImage, beta, 0.0)
            # new_frame = np.zeros(frame.shape, np.uint8)
            # for i, contour in enumerate(contours):
            #     c_area = cv2.contourArea(contour)
            #     mask = np.zeros(frame.shape, np.uint8)
            #     new_frame = cv2.drawContours(mask, contours, i, 255, cv2.FILLED)
            #     mask = cv2.bitwise_and(frame, new_frame)
            #     new_frame = cv2.bitwise_or(mask, backgroundImage)
            # dst = new_frame

            # Import background image            
            # alpha = 0.75
            # beta = (1.0 - alpha)
            
            #dst = cv2.addWeighted(dst, alpha, frame, beta, 0.0)

            # dst = cv2.addWeighted(backgroundImage, alpha, dst, beta, 0.0)
            #dst = cv2.addWeighted(grayFrame, alpha, dst, beta, 0.0)
            cv2.imshow(window_name, dst)
        success, frame = cameraCapture.read()

    cv2.destroyWindow(window_name)


window_name = 'Web Cam'
display_web_camera(window_name)

