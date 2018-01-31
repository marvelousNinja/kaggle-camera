import os
from multiprocessing.pool import ThreadPool
from functools import partial
import itertools

from tqdm import tqdm
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

from camera.utils import pipe, encode_labels, in_batches
from camera.data import list_all_samples_in, train_test_holdout_split, list_dirs_in
from camera.data import read_jpeg

from camera.transforms import (
    default_transforms_and_weights, random_transform,
    crop_center, crop_random,
    spam_11_3, spam_11_5, spam_14_edge,
    identity
)

from camera.networks import densenet_40, densenet_121, densenet_169, densenet_201

def read_jpeg_cached(cache, crop_size, path):
    image = cache.get(path)
    if image is not None:
        return image
    else:
        image = np.array(crop_center(crop_size, read_jpeg(path)))
        cache[path] = image
        return image

def infinite_generator(initializer):
    while True:
        generator = initializer()
        for element in generator:
            yield element

def conduct(
        data_dir,
        learning_rate,
        crop_size,
        n_epochs,
        batch_size,
        outer_crop_strategy,
        inner_crop_strategy,
        transform_strategy,
        residual_filter_strategy,
        overfit_run,
        network
    ):

    crop_strategies = {
        'none': identity,
        'crop_center': crop_center,
        'crop_random': crop_random
    }

    filter_strategies = {
        'none': identity,
        'spam_11_3': spam_11_3,
        'spam_11_5': spam_11_5,
        'spam_14_edge': spam_14_edge
    }

    transform_strategies = {
        'none': identity,
        'random': partial(random_transform, default_transforms_and_weights())
    }

    networks = {
        'densenet_40': densenet_40,
        'densenet_121': densenet_121,
        'densenet_169': densenet_169,
        'densenet_201': densenet_201
    }

    outer_crop_size = crop_size * 2 + 8

    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    all_labels = list_dirs_in(os.path.join(data_dir, 'train'))
    train, test, _ = train_test_holdout_split(all_samples)

    if overfit_run:
        train = train[:batch_size]
        test = test[:batch_size]

    n_batches = int(np.ceil(len(train) / batch_size))

    cache = dict()
    train_pipeline = partial(pipe, [
        partial(read_jpeg_cached, cache, outer_crop_size),
        partial(crop_strategies[outer_crop_strategy], outer_crop_size),
        transform_strategies[transform_strategy],
        filter_strategies[residual_filter_strategy],
        partial(crop_strategies[inner_crop_strategy], crop_size)
    ])

    test_pipeline = partial(pipe, [
        partial(read_jpeg_cached, cache, outer_crop_size),
        partial(crop_center, outer_crop_size),
        transform_strategies[transform_strategy],
        filter_strategies[residual_filter_strategy],
        partial(crop_center, crop_size)
    ])

    pool = ThreadPool(initializer=np.random.seed)

    label_encoder = partial(encode_labels, all_labels)
    to_batch = partial(in_batches, batch_size)

    paths, labels = zip(*test)
    x_test = np.stack(list(tqdm(pool.imap(test_pipeline, paths))))
    y_test = np.stack(label_encoder(labels))

    input_shape = (crop_size, crop_size, 3)
    num_classes = len(all_labels)
    model = networks[network](input_shape, num_classes)
    optimizer = Adam(learning_rate)
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

    print(model.summary())

    def train_generator_initializer():
        np.random.shuffle(train)
        paths, labels = zip(*train)
        labels = label_encoder(labels)
        return zip(to_batch(pool.imap(train_pipeline, paths)), to_batch(labels))

    reduce_lr = ReduceLROnPlateau(patience=5, verbose=1)

    model.fit_generator(
        generator=infinite_generator(train_generator_initializer),
        steps_per_epoch=n_batches,
        epochs=n_epochs,
        verbose=2,
        callbacks=[reduce_lr],
        validation_data=(x_test, y_test),
        initial_epoch=0
    )
