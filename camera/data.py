import os
import glob
import numpy as np
import jpeg4py as jpeg
import cv2

def read_png(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def read_jpeg(path):
    return jpeg.JPEG(str(path)).decode()

def read_jpeg_cached(cache, preprocess, path):
    image = cache.get(path)
    if image is not None:
        return image
    else:
        image = preprocess(read_jpeg(path))
        cache[path] = image
        return image

def list_images_in(path):
    extensions = ['jpg', 'JPG', 'tif', 'png', 'jpg.*']
    files = []
    for extension in extensions:
        files.extend(glob.glob(path + f'/*.{extension}'))
    return np.sort(files)

def list_dirs_in(path):
    dirs = [dir for dir in os.listdir(path) if not dir.startswith('.')]
    return np.sort(dirs)

def label_mapping():
    return {
        'HTC-1-M7': 0,
        'iPhone-4s': 1,
        'Motorola-Droid-Maxx': 2,
        'Samsung-Galaxy-Note3': 3,
        'Samsung-Galaxy-S4': 4,
        'LG-Nexus-5x': 5,
        'Motorola-Nexus-6': 6,
        'iPhone-6': 7,
        'Motorola-X': 8,
        'Sony-NEX-7': 9
    }

def inverse_label_mapping():
    mapping = label_mapping()
    return { v: k for k, v in mapping.items() }

def list_all_samples_in(path):
    image_paths_and_labels = list()
    mapping = label_mapping()

    for label in list_dirs_in(path):
        image_paths = list_images_in(os.path.join(path, label))
        image_paths_and_labels.extend(map(lambda path: [path, mapping[label]], image_paths))

    return image_paths_and_labels

def train_test_holdout_split(samples, seed=11):
    np.random.seed(seed)
    shuffled_samples = list(samples)
    np.random.shuffle(shuffled_samples)
    sample_count = len(samples)
    train = shuffled_samples[0:int(sample_count * 0.7)]
    test = shuffled_samples[int(sample_count * 0.7):int(sample_count * 0.85)]
    holdout = shuffled_samples[int(sample_count * 0.85):]

    print('First 5 holdout samples...')
    print(np.array(holdout[:5]))
    return train, test, holdout

def get_datasets(data_dir):
    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    return train_test_holdout_split(all_samples)

def get_test_dataset(data_dir):
    return list_images_in(os.path.join(data_dir, 'test'))
