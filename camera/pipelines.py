from functools import partial
import numpy as np
from camera.utils import pipe, evolve_at, transform_to_sample_weight
from camera.data import read_jpeg, read_jpeg_cached, read_png
from camera.transforms import crop_center, random_transform, default_transforms_and_weights, crop_random, image_filters, random_flip, spam_11_5

def training_pipeline(cache, image_filter, allow_flips, allow_weights, crop_size, record):
    path, label = record
    outer_crop_size = 780
    image = read_jpeg_cached(cache, partial(crop_center, outer_crop_size), path)
    if allow_flips: image = random_flip(image)
    image, transform_name = random_transform(default_transforms_and_weights(), image)
    image = crop_random(crop_size, image)
    image = image_filters()[image_filter](image.astype(np.float32))
    sample_weight = transform_to_sample_weight(transform_name)
    return [image, label, sample_weight if allow_weights else 1.0]

def validation_pipeline(image_filter, allow_weights, crop_size, record):
    path, label = record
    outer_crop_size = 512
    image = read_jpeg(path)
    image = crop_center(outer_crop_size, image)
    image, transform_name = random_transform(default_transforms_and_weights(), image)
    image = crop_center(crop_size, image)
    image = image_filters()[image_filter](image.astype(np.float32))
    sample_weight = transform_to_sample_weight(transform_name)
    return [image, label, sample_weight if allow_weights else 1.0]

def submission_pipeline(image_filter, crop_size, path):
    image = read_png(path)
    image = crop_center(crop_size, image)
    image = image_filters()[image_filter](image.astype(np.float32))
    return image
