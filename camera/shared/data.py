import os
import glob
from skimage import io

def load_sample_image():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return io.imread(dir_path + '/train/iPhone-6/(iP6)1.jpg')

def get_all_labels():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/train'
    return [dir for dir in os.listdir(dir_path) if not dir.startswith('.')]

def get_image_paths(label):
    pattern = os.path.dirname(os.path.realpath(__file__)) + '/train/{}/*.[jJ][pP][gG]'.format(label)
    return glob.glob(pattern)
