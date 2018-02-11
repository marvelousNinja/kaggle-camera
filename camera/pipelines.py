"""Data processing pipelines"""
from functools import partial

import numpy as np

from camera.data import read_image
from camera.data import read_image_cached
from camera.transforms import crop_center
from camera.transforms import crop_random
from camera.transforms import crop_top_left
from camera.transforms import crop_top_right
from camera.transforms import crop_bottom_left
from camera.transforms import crop_bottom_right
from camera.transforms import default_transforms_and_weights
from camera.transforms import image_filters
from camera.transforms import random_flip
from camera.transforms import random_transform
from camera.utils import transform_to_sample_weight

def training_pipeline(cache, image_filter, allow_flips, allow_weights, crop_size, record):
    path, label = record
    outer_crop_size = 780
    image = read_image_cached(cache, partial(crop_center, outer_crop_size), path)
    if allow_flips: image = random_flip(image)
    image, transform_name = random_transform(default_transforms_and_weights(), image)
    image = crop_random(crop_size, image)
    image = image_filters()[image_filter](image.astype(np.float32))
    sample_weight = transform_to_sample_weight(transform_name)
    return [image, label, sample_weight if allow_weights else 1.0]

def validation_pipeline(image_filter, allow_weights, crop_size, record):
    path, label = record
    outer_crop_size = 512
    image = read_image(path)
    image = crop_center(outer_crop_size, image)
    image, transform_name = random_transform(default_transforms_and_weights(), image)
    image = crop_center(crop_size, image)
    image = image_filters()[image_filter](image.astype(np.float32))
    sample_weight = transform_to_sample_weight(transform_name)
    return [image, label, sample_weight if allow_weights else 1.0]

def tta_pipeline(image_filter, allow_weights, crop_size, record):
    path, label = record
    outer_crop_size = 512
    image = read_image(path)
    image = crop_center(outer_crop_size, image)
    image, transform_name = random_transform(default_transforms_and_weights(), image)
    sample_weight = transform_to_sample_weight(transform_name)
    tta_crops = get_tta_crops(crop_size, image)
    tta_crops = list(map(image_filters()[image_filter], tta_crops))
    return [tta_crops, label, sample_weight if allow_weights else 1.0]

def tta_submission_pipeline(image_filter, crop_size, path):
    image = read_image(path)
    tta_crops = get_tta_crops(crop_size, image)
    tta_crops = list(map(image_filters()[image_filter], tta_crops))
    return tta_crops

def get_tta_crops(crop_size, image):
    image = image.astype(np.float32)
    flipped_image = np.fliplr(image)

    return [
        crop_top_left(crop_size, image),
        crop_top_right(crop_size, image),
        crop_bottom_left(crop_size, image),
        crop_bottom_right(crop_size, image),
        crop_center(crop_size, image),
        crop_top_left(crop_size, flipped_image),
        crop_top_right(crop_size, flipped_image),
        crop_bottom_left(crop_size, flipped_image),
        crop_bottom_right(crop_size, flipped_image),
        crop_center(crop_size, flipped_image),
    ]
