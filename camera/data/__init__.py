import os
import glob
from skimage import io

def load_sample_image(data_dir):
    return io.imread(data_dir + '/train/iPhone-6/(iP6)1.jpg')

def get_all_labels(data_dir):
    return [dir for dir in os.listdir(data_dir + '/train') if not dir.startswith('.')]

def get_image_paths(label, data_dir):
    pattern = data_dir + '/train/{}/*.[jJ][pP][gG]'.format(label)
    return glob.glob(pattern)
