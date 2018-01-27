import os
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools

import jpeg4py as jpeg
from tqdm import tqdm
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D, Convolution2D, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2
from keras.utils import to_categorical
from keras_contrib.applications import DenseNet
from keras import backend as K

from camera.data import list_all_samples_in, train_test_holdout_split, list_dirs_in
from camera.transforms import default_transforms_and_weights
from camera.transforms import random_transform

def read_jpeg(path):
    return jpeg.JPEG(str(path)).decode()

def crop_center(size, image):
    top_x = image.shape[0] // 2 - size // 2
    top_y = image.shape[1] // 2 - size // 2
    return image[top_x:top_x + size, top_y:top_y + size]

def pipe(funcs, target):
    result = target
    for func in funcs:
        result = func(result)
    return result

def apply_filter(image_filter, image):
    return cv2.filter2D(image.astype(np.float), -1, image_filter)

def in_batches(batch_size, iterable):
    batch = list()
    for element in iterable:
        batch.append(element)
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = list()
    if len(batch) > 0:
        yield np.stack(batch)

def encode_labels(all_labels, labels):
    mapped_labels = np.array(labels)
    for code, label in enumerate(all_labels):
        mapped_labels[mapped_labels == label] = code
    return to_categorical(mapped_labels, num_classes=len(all_labels))

def learning_schedule(epoch):
    return 0.015 / (2 ** (epoch // 10))

def drop_n_and_freeze(n, model):
    for _ in range(n):
        model.layers.pop()

    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]

    for layer in model.layers:
        layer.trainable = False

    return model

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).reshape(image.shape[0], image.shape[1], 1)

def reshape(image):
    return image.reshape(image.shape[0], image.shape[1], 1)

def crop_random(size, image):
    top_x = np.random.randint(image.shape[0] - size)
    top_y = np.random.randint(image.shape[1] - size)
    return image[top_x:top_x + size, top_y:top_y + size]

def conduct(data_dir):
    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    all_labels = list_dirs_in(os.path.join(data_dir, 'train'))
    train, test, _ = train_test_holdout_split(all_samples)

    crop_size = 32

    image_filter = np.array([
        [-1, +2, -2, +2, -1],
        [+2, -6, +8, -6, +2],
        [-2, +8, -12, +8, -2],
        [+2, -6, +8, -6, +2],
        [-1, +2, -2, +2, -1]
    ]) / 12

    train_pipeline = partial(pipe, [
        read_jpeg,
        partial(crop_center, 512),
        partial(random_transform, default_transforms_and_weights()),
        partial(apply_filter, image_filter),
        partial(crop_random, crop_size)
    ])

    test_pipeline = partial(pipe, [
        read_jpeg,
        partial(crop_center, 512),
        partial(random_transform, default_transforms_and_weights()),
        partial(apply_filter, image_filter),
        partial(crop_center, crop_size)
    ])

    pool = Pool(processes=cpu_count() - 2, initializer=np.random.seed)

    n_epochs = 50
    batch_size = 64
    n_batches = len(train) // batch_size
    label_encoder = partial(encode_labels, all_labels)
    to_batch = partial(in_batches, batch_size)

    paths, labels = zip(*test)
    x_test = np.stack(list(tqdm(pool.imap(test_pipeline, paths))))
    y_test = np.stack(label_encoder(labels))

    model = DenseNet(
        depth=40,
        nb_dense_block=3,
        growth_rate=12,
        nb_filter=16,
        dropout_rate=0.0,
        input_shape=(32, 32, 3),
        pooling='avg',
        include_top=True,
        weights=None
    )

    model.compile(
        optimizer=Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    for epoch in tqdm(range(n_epochs)):
        np.random.shuffle(train)
        paths, labels = zip(*train)
        labels = label_encoder(labels)

        for x_train, y_train in tqdm(zip(to_batch(pool.imap(train_pipeline, paths)), to_batch(labels)), total=n_batches):
            model.train_on_batch(x_train, y_train)

        train_metrics = model.test_on_batch(x_train, y_train)
        test_metrics = model.test_on_batch(x_test, y_test)
        tqdm.write('Training ' + str(list(zip(model.metrics_names, train_metrics))) + ' Validation ' + str(list(zip(model.metrics_names, test_metrics))))
