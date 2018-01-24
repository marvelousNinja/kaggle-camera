import os
import glob
import numpy as np

def list_images_in(path):
    extensions = ['jpg', 'JPG', 'tiff', 'png']
    files = []
    for extension in extensions:
        files.extend(glob.glob(path + f'/*.{extension}'))
    return files

def list_dirs_in(path):
    return [dir for dir in os.listdir(path) if not dir.startswith('.')]

def list_all_samples_in(path):
    image_paths_and_labels = []

    for label in list_dirs_in(path):
        image_paths = list_images_in(os.path.join(path, label))
        labels = [label] * len(image_paths)
        image_paths_and_labels.extend(zip(image_paths, labels))

    return image_paths_and_labels

def train_test_holdout_split(samples, seed=11):
    np.random.seed(seed)
    shuffled_samples = np.array(samples)
    np.random.shuffle(shuffled_samples)
    sample_count = len(samples)
    train = shuffled_samples[0:int(sample_count * 0.7)]
    test = shuffled_samples[int(sample_count * 0.7):int(sample_count * 0.85)]
    holdout = shuffled_samples[int(sample_count * 0.85):]
    return train, test, holdout
