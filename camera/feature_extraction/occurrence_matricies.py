import numpy as np
from camera.interpolation.metrics import difference
from camera.interpolation.nearest_neighbour import interpolate_bayer

def occurrence_matrix(image):
    interpolated_image = interpolate_bayer(image)
    threshold = 2
    # TODO AS: Since diff is later used as index, make sure it is >= 0
    diff = difference(image, interpolated_image, quantization=2, threshold=threshold) + threshold
    red, green, blue = diff[::, ::, 0], diff[::, ::, 1], diff[::, ::, 2]
    pattern_dim = 2 * threshold + 1
    patterns = np.zeros((pattern_dim, pattern_dim, pattern_dim))
    np.add.at(patterns, [red[::2, ::2], red[::2, 1::2], red[1::2, 1::2]], 1)
    patterns /= (red.shape[0] * red.shape[1] / 4)
    return patterns
