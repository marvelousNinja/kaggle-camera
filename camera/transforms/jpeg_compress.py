import cv2

def jpeg_compress(quality, image):
    _, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.cvtColor(cv2.imdecode(encoded_image, 1), cv2.COLOR_BGR2RGB)
