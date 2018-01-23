import os
from multiprocessing import Queue, Manager
import numpy as np
import cv2
from camera.data import list_dirs_in
from camera.data import list_images_in
from camera.transforms import adjust_gamma
from camera.transforms import flip_horizontally
from camera.transforms import jpeg_compress
from camera.transforms import resize

def intensity_texture_score(image):
    image = image / 255
    flat_image = image.reshape(-1, 3)
    channel_mean = flat_image.mean(axis=0)
    channel_std = flat_image.std(axis=0)

    channel_mean_score = -4 * channel_mean ** 2 + 4 * channel_mean
    channel_std_score = 1 - np.exp(-2 * np.log(10) * channel_std)

    channel_mean_score_aggr = channel_mean_score.mean()
    channel_std_score_aggr = channel_std_score.mean()
    return 0.7 * channel_mean_score_aggr + 0.3 * channel_std_score_aggr

def list_all_samples_in(dir):
    labels_and_image_paths = []

    for label in list_dirs_in(dir):
        image_paths = list_images_in(os.path.join(dir, label))
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
    threshold = 0.5
    patches = []
    for i in range(image.shape[0] // size):
        for j in range(image.shape[1] // size):
            patches.append(image[i * size:(i + 1) * size, j * size:(j + 1) * size])

    patch_scores = []
    for patch in patches:
        patch_scores.append(intensity_texture_score(patch))

    scored_score_indicies = np.argsort(patch_scores)[::-1]
    sorted_scores = np.array(patch_scores)[scored_score_indicies]
    sorted_patches = np.array(patches)[scored_score_indicies]
    return sorted_patches[sorted_scores >= threshold][:limit]

def random_crop(image, size, limit):
    x_max, y_max = image.shape[0] - size, image.shape[1] - size
    for _ in range(limit):
        x, y = np.random.randint(0, x_max), np.random.randint(0, y_max)
        yield image[x:x + size, y:y + size]

def process_label_image_path_pair(label, image_path, crop_generator, crop_size, crop_limit, queue):
    transforms = {
        'unalt': lambda image: image,
        'gamma80': lambda image: adjust_gamma(image, 0.8),
        'gamma120': lambda image: adjust_gamma(image, 1.2),
        'jpeg70': lambda image: jpeg_compress(image, 70),
        'jpeg90': lambda image: jpeg_compress(image, 90),
        'resize50': lambda image: resize(image, 0.5),
        'resize80': lambda image: resize(image, 0.8),
        'resize150': lambda image: resize(image, 1.5),
        'resize200': lambda image: resize(image, 2.0)
    }

    transform_weights = np.array([9, 1, 1, 1, 1, 1, 1, 1, 1])
    transform_weights = transform_weights / sum(transform_weights)
    transform_names = list(transforms.keys())

    image_id = os.path.splitext(os.path.basename(image_path))[0]
    transform_name = np.random.choice(transform_names, p=transform_weights)
    transform = transforms[transform_name]
    transformed_image = cv2.cvtColor(transform(cv2.imread(image_path)), cv2.COLOR_BGR2RGB)
    # image_id, label, transform_name, image
    for patch in crop_generator(transformed_image, crop_size, crop_limit):
        queue.put((image_id, label, transform_name, patch))

def parallel_image_generator(pool, labels_and_image_paths, crop_generator=random_crop, crop_size=512, crop_limit=15):
    manager = Manager()
    queue = manager.Queue(1000)

    for label, image_path in labels_and_image_paths:
        pool.apply_async(process_label_image_path_pair, (label, image_path, crop_generator, crop_size, crop_limit, queue))

    while True:
        yield queue.get()
