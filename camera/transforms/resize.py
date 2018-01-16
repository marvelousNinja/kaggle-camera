import cv2

# 0.5, 0.8, 1.5, 2.0
def resize(ratio):
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
