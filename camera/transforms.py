"""Image transform utils"""
from functools import partial

import cv2
import numpy as np
import keras.applications

def identity(image):
    return image

def resize(ratio, image):
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

def jpeg_compress(quality, image):
    _, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.cvtColor(cv2.imdecode(encoded_image, 1), cv2.COLOR_BGR2RGB)

def adjust_gamma(gamma, image):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)

def default_transforms_and_weights():
    return (
        (identity, 8),
        (partial(adjust_gamma, 0.8), 1),
        (partial(adjust_gamma, 1.2), 1),
        (partial(jpeg_compress, 70), 1),
        (partial(jpeg_compress, 90), 1),
        (partial(resize, 0.5), 1),
        (partial(resize, 0.8), 1),
        (partial(resize, 1.5), 1),
        (partial(resize, 2.0), 1)
    )

def random_transform(transforms_and_weights, image):
    transforms, weights = zip(*transforms_and_weights)
    weights = np.array(weights)
    probabilities = weights / sum(weights)
    transform = np.random.choice(transforms, p=probabilities)
    return [transform(image), getattr(transform, '__name__', 'unknown')]

def image_filters():
    return {
        'densenet': keras.applications.densenet.preprocess_input,
        'resnet50': keras.applications.resnet50.preprocess_input,
        'mobilenet': keras.applications.mobilenet.preprocess_input,
    }

def random_flip(image):
    if np.random.rand() < 0.5:
        if np.random.rand() < 0.5:
            image = np.fliplr(image)

        if np.random.rand() < 0.5:
            image = np.rot90(image)

        if np.random.rand() < 0.5:
            image = np.flipud(image)

    return image

def crop_random(size, image):
    top_x = np.random.randint(image.shape[0] - size)
    top_y = np.random.randint(image.shape[1] - size)
    return np.array(image[top_x:top_x + size, top_y:top_y + size])

def crop_center(size, image):
    top_x = image.shape[0] // 2 - size // 2
    top_y = image.shape[1] // 2 - size // 2
    return np.array(image[top_x:top_x + size, top_y:top_y + size])

def crop_top_left(crop_size, image):
    return np.array(image[0:crop_size, 0:crop_size])

def crop_top_right(crop_size, image):
    return np.array(image[0:crop_size, -crop_size:])

def crop_bottom_left(crop_size, image):
    return np.array(image[-crop_size:, 0:crop_size])

def crop_bottom_right(crop_size, image):
    return np.array(image[-crop_size:, -crop_size:])
