import numpy as np
from camera.interpolation.metrics import difference
from camera.interpolation.nearest_neighbour import interpolate_bayer

def occurrence_matrix(image, quantization=2, threshold=3):
    # TODO AS: We need interpolation for red channel only
    interpolated_image = interpolate_bayer(image)
    # TODO AS: Since diff is later used as index, make sure it is >= 0
    # TODO AS: We need difference for red channel only
    diff = difference(image, interpolated_image, quantization=quantization, threshold=threshold) + threshold
    red = diff[::, ::, 0]
    # green = diff[::, ::, 1]
    # blue = diff[::, ::, 2]
    pattern_dim = 2 * threshold + 1
    patterns = np.zeros((pattern_dim, pattern_dim, pattern_dim))
    np.add.at(patterns, [red[::2, ::2], red[::2, 1::2], red[1::2, 1::2]], 1)
    patterns /= (red.shape[0] * red.shape[1] / 4)
    return patterns
