import numpy as np

def flip_horizontally(image):
    # Drop first column to preserve Bayer pattern
    return np.array(np.rflip(image)[:, 1:])
