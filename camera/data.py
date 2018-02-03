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
    extensions = ['jpg', 'JPG', 'tif', 'png']
    files = []
    for extension in extensions:
        files.extend(glob.glob(path + f'/*.{extension}'))
    return files

def list_dirs_in(path):
    return [dir for dir in os.listdir(path) if not dir.startswith('.')]

def label_mapping(path):
    all_labels = list_dirs_in(path)
    mapping = dict()

    for i, label in enumerate(all_labels):
        mapping[label] = i

    return mapping

def inverse_label_mapping(path):
    mapping = label_mapping(path)
    return { v: k for k, v in mapping.items() }

def list_all_samples_in(path):
    image_paths_and_labels = list()
    mapping = label_mapping(path)

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
    return train, test, holdout

def get_datasets(data_dir):
    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    return train_test_holdout_split(all_samples)

def get_test_dataset(data_dir):
    return list_images_in(os.path.join(data_dir, 'test'))
