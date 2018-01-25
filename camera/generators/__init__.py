import os
from multiprocessing import Pool
import numpy as np
import cv2

def process_image(path, label, transform, crop):
    image = cv2.imread(path)
    transformed_image = transform(image)
    rgb_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    return crop(rgb_image), label

def process_image_star(args):
    return process_image(*args)

def image_generator(image_paths_and_labels, transform, crop, loop=True, seed=11, pool=Pool(processes=None, initializer=np.random.seed)):
    np.random.seed(seed)

    while True:
        pairs = np.array(image_paths_and_labels)
        np.random.shuffle(pairs)

        paths, labels = zip(*image_paths_and_labels)
        params = zip(
            paths,
            labels,
            [transform] * len(paths),
            [crop] * len(paths)
        )

        for patches, label in pool.imap(process_image_star, params):
            for patch in patches:
                yield patch, label

        if not loop:
            break

def in_batches(batch_size, iterable):
    feature_batch = list()
    label_batch = list()

    for features, label in iterable:
        feature_batch.append(features)
        label_batch.append(label)

        if len(feature_batch) == batch_size:
            yield (np.array(feature_batch), np.array(label_batch))
            feature_batch = list()
            label_batch = list()

    yield (np.array(feature_batch), np.array(label_batch))
