import numpy as np

def difference(one, another, quantization, threshold):
    # TODO AS: Not sure about //
    diff = (one.astype(np.int) - another.astype(np.int)) // quantization
    return np.clip(diff, a_min=-threshold, a_max=threshold)
