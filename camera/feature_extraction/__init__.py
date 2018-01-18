import numpy as np

def intra_channel(image, threshold=3, quantization=2):
    # 1. Interpolate
    red_image = image[:, :, 0].astype(np.int16)
    interpolated_image = np.array(red_image)
    true_red = red_image[1::2, 0::2]
    interpolated_image[0::2, 0::2] = true_red
    interpolated_image[1::2, 1::2] = true_red
    interpolated_image[0::2, 1::2] = true_red

    # 2. Calculate difference, quantize and clip
    diff = red_image - interpolated_image
    diff //= quantization
    # TODO AS: Performance eater
    np.clip(diff, a_min=-threshold, a_max=threshold, out=diff)
    diff += threshold

    # 3. Calculate co-occurrence matrix
    residual_dimension = 2 * threshold + 1
    patterns = np.zeros((residual_dimension, residual_dimension, residual_dimension), dtype=np.int)
    np.add.at(patterns, [diff[::2, ::2], diff[::2, 1::2], diff[1::2, 1::2]], 1)
    patterns = patterns / (diff.shape[0] * diff.shape[1] / 4)
    return patterns.reshape(-1)

def inter_channel(image, threshold=3, quantization=2):
    # 1. Interpolate
    red_green_image = image[:, :, :2].astype(np.int32)
    interpolated_image = np.array(red_green_image)

    true_red = red_green_image[1::2, 0::2, 0]
    interpolated_image[0::2, 0::2, 0] = true_red
    interpolated_image[1::2, 1::2, 0] = true_red
    interpolated_image[0::2, 1::2, 0] = true_red

    true_green_1 = red_green_image[0::2, 0::2, 1]
    true_green_2 = red_green_image[1::2, 1::2, 1]
    interpolated_image[0::2, 1::2, 1] = true_green_1
    interpolated_image[1::2, 0::2, 1] = true_green_2

    # 2. Calculate difference, quantize and clip
    diff = red_green_image - interpolated_image
    diff //= quantization
    # TODO AS: Performance eater
    np.clip(diff, a_min=-threshold, a_max=threshold, out=diff)
    diff += threshold

    # 3. Calculate co-occurrence matrix
    residual_dimension = 2 * threshold + 1
    patterns = np.zeros((residual_dimension, residual_dimension, residual_dimension), dtype=np.int)
    np.add.at(patterns, [diff[::2, ::2, 0], diff[::2, 1::2, 0], diff[0::2, 1::2, 1]], 1)
    np.add.at(patterns, [diff[1::2, 1::2, 0], diff[::2, 1::2, 0], diff[0::2, 1::2, 1]], 1)
    patterns = patterns / (diff.shape[0] * diff.shape[1] / 4)
    return patterns.reshape(-1)

def gram_matrix(image):
    image = image[0:64, 0:64, :]
    return (image * np.transpose(image, axes=(1, 0, 2))).reshape(-1)
