import numpy as np

def difference(one, another, quantization, threshold):
    # TODO AS: Not sure about //
    diff = (one - another) // quantization
    return np.clip(diff, a_min=-threshold, a_max=threshold)
