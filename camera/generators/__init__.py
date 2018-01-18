import numpy as np
import cv2
import os
from camera.data import list_dirs_in
from camera.data import list_images_in
from camera.transforms import adjust_gamma
from camera.transforms import flip_horizontally
from camera.transforms import jpeg_compress
from camera.transforms import resize

def list_all_samples_in(data_dir):
    labels_and_image_paths = []

    for label in list_dirs_in(data_dir + '/train'):
        image_paths = list_images_in(data_dir + '/train/' + label)
        labels = [label] * len(image_paths)
        labels_and_image_paths.extend(zip(labels, image_paths))

    return labels_and_image_paths

def train_test_holdout_split(samples, seed=11):
    np.random.seed(seed)
    shuffled_samples = np.array(samples)
    np.random.shuffle(shuffled_samples)
    sample_count = len(samples)
    train = shuffled_samples[0:int(sample_count * 0.7)]
    test = shuffled_samples[int(sample_count * 0.7):int(sample_count * 0.85)]
    holdout = shuffled_samples[int(sample_count * 0.85):]
    return train, test, holdout

def center_crop(image, size, _):
    center_x = image.shape[0] // 2 - 1
    center_y = image.shape[1] // 2 - 1
    top_x, top_y = center_x - size // 2, center_y - size // 2
    return [image[top_x:top_x + size, top_y:top_y + size]]

def sequential_crop(image, size, limit):
    for i in range(image.shape[0] // size):
        for j in range(image.shape[1] // size):
            if (i * (image.shape[1] // size) + j) >= limit:
                return

            yield image[i * size:(i + 1) * size, j * size:(j + 1) * size]

def random_crop(image, size, limit):
    x_max, y_max = image.shape[0] - size, image.shape[1] - size
    for _ in range(limit):
        x, y = np.random.randint(0, x_max), np.random.randint(0, y_max)
        yield image[x:x + size, y:y + size]

def image_generator(labels_and_image_paths, crop_generator=random_crop, crop_size=512, seed=11):
    transforms = {
        'unalt': lambda image: image,
        # 'gamma80': lambda image: adjust_gamma(image, 0.8),
        # 'gamma120': lambda image: adjust_gamma(image, 1.2),
        # 'jpeg70': lambda image: jpeg_compress(image, 70),
        # 'jpeg90': lambda image: jpeg_compress(image, 90),
        # 'resize50': lambda image: resize(image, 0.5),
        # 'resize80': lambda image: resize(image, 0.8),
        # 'resize150': lambda image: resize(image, 1.5),
        # 'resize200': lambda image: resize(image, 2.0)
    }

    # transform_weights = np.array([9, 1, 1, 1, 1, 1, 1, 1, 1])
    transform_weights = np.array([1])
    transform_weights = transform_weights / sum(transform_weights)
    transform_names = list(transforms.keys())

    for label, image_path in labels_and_image_paths:
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        transform_name = np.random.choice(transform_names, p=transform_weights)
        transform = transforms[transform_name]
        transformed_image = cv2.cvtColor(transform(cv2.imread(image_path)), cv2.COLOR_BGR2RGB)
        # image_id, label, altered, image
        for patch in crop_generator(transformed_image, crop_size, 10):
            yield image_id, label, transform_name, patch
