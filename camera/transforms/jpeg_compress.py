import cv2

def jpeg_compress(image):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encoded_image = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    decoded_image = cv2.imdecode(encoded_image, 1)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
