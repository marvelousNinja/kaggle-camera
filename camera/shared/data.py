import os
from skimage import io

def load_sample_image():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return io.imread(dir_path + '/train/iPhone-6/(iP6)1.jpg')
