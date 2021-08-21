import cv2

cap = cv2.VideoCapture(0)

sub = cv2.createBackgroundSubtractorKNN(
    history=30, dist2Threshold=70, detectShadows=True)

while cap.isOpened():
    ret, fg_img = cap.read()

    if not ret:
        break


    mask = sub.apply(fg_img)

    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    #result = cv2.bitwise_and(bg_img, fg_img, mask=mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(mask, fg_img)
    # cv2.imshow('fg', fg_img)
    # cv2.imshow('bg', bg_img)
    # cv2.imshow('mask', mask)
    cv2.imshow('result', result)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
