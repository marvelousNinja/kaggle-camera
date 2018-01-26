import os
from multiprocessing import Pool, cpu_count
import numpy as np
import jpeg4py as jpeg

def process_image(path, label, transform, crop):
    # TODO AS: Use cv2 for PNG
    image = jpeg.JPEG(str(path)).decode()
    transformed_image = transform(image)
    return crop(transformed_image), label

def process_image_star(args):
    return process_image(*args)

def image_generator(image_paths_and_labels, transform, crop, loop=True, seed=11, pool=Pool(processes=cpu_count() - 2, initializer=np.random.seed)):
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
