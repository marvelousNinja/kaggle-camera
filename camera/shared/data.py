import os
from skimage import io

def load_sample_image():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return io.imread(dir_path + '/train/iPhone-6/(iP6)1.jpg')

def load_images(label):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return io.imread_collection(dir_path + '/train/{}/*.[jJ][pP][gG]'.format(label))

def get_all_labels():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/train'
    return [dir for dir in os.listdir(dir_path) if not dir.startswith('.')]
