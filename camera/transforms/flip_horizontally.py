import numpy as np

def flip_horizontally(image):
    # TODO AS: Drop first column to preserve Bayer pattern?
    return np.array(np.fliplr(image))
