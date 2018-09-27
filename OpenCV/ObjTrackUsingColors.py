import cv2
import numpy as np

def get_frame(cap, scaling_factor):
    _, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor,
                       interpolation=cv2.INTER_AREA)
    return  frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.9

    while True:
        frame = get_frame(cap, scaling_factor)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 70, 80])
        upper = np.array([50, 250, 255])

        mask = cv2.inRange(hsv, lower, upper)
        img_bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)
        img_median_blurred = cv2.medianBlur(img_bitwise_and, 5)

        cv2.imshow('Input', frame)
        cv2.imshow('Output', img_median_blurred)

        c = cv2.waitKey(5)
        if c == 27:
            break

    cv2.destroyAllWindows()
