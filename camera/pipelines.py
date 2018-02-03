from functools import partial
import numpy as np
from camera.utils import pipe, evolve_at, transform_to_sample_weight
from camera.data import read_jpeg, read_jpeg_cached
from camera.transforms import crop_center, random_transform, default_transforms_and_weights, crop_random, image_filters

def validation_pipeline(crop_size, image_filter, apply_transforms, apply_flips, calculate_weights, record):
    path = record[0]
    label = record[1]

    outer_crop_size = crop_size * 2 + 8

    image = read_jpeg(path)
    image = crop_center(outer_crop_size, image)

    if apply_flips:
        if np.random.rand() > 0.5:
            image = np.fliplr(image)

        if np.random.rand() > 0.5:
            image = np.rot90(image)

    if apply_transforms:
        image, transform_name = random_transform(default_transforms_and_weights(), image)
    else:
        transform_name = 'identity'

    image = crop_random(crop_size, image)
    image = image_filters()[image_filter](image)

    if calculate_weights:
        sample_weight = transform_to_sample_weight(transform_name)
    else:
        sample_weight = 1

    return [image, label, sample_weight]

def train_pipeline(cache, crop_size, image_filter, apply_transforms, apply_flips, calculate_weights, record):
    path = record[0]
    label = record[1]

    outer_crop_size = crop_size * 2 + 8

    image = read_jpeg_cached(cache, partial(crop_center, outer_crop_size), path)

    if apply_flips:
        if np.random.rand() > 0.5:
            image = np.fliplr(image)

        if np.random.rand() > 0.5:
            image = np.rot90(image)

    if apply_transforms:
        image, transform_name = random_transform(default_transforms_and_weights(), image)
    else:
        transform_name = 'identity'

    image = crop_random(crop_size, image)
    image = image_filters()[image_filter](image)

    if calculate_weights:
        sample_weight = transform_to_sample_weight(transform_name)
    else:
        sample_weight = 1

    return [image, label, sample_weight]
