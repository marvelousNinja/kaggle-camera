import os
from multiprocessing import Pool
from functools import partial
from itertools import islice
import numpy as np
import cv2
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D, Convolution2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2
from keras.utils import to_categorical
from keras import backend as K
from camera.data import list_dirs_in, list_all_samples_in, train_test_holdout_split
from camera.transforms import default_transforms_and_weights
from camera.transforms import random_transform
from camera.transforms import sequential_crop
from camera.transforms import center_crop
from camera.generators import image_generator
from camera.generators import in_batches
from camera.transforms import identity

def learning_schedule(epoch):
    return 0.015 / (1 + epoch // 10)

def build_cnn():
    reg_coef = 0.00075

    cnn = Sequential([
        Convolution2D(filters=32, kernel_size=4, strides=1, padding='valid', input_shape=(64, 64, 3), kernel_regularizer=l2(reg_coef)),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        Convolution2D(filters=48, kernel_size=5, strides=1, padding='valid', kernel_regularizer=l2(reg_coef)),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        Convolution2D(filters=64, kernel_size=5, strides=1, padding='valid', kernel_regularizer=l2(reg_coef)),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        Convolution2D(filters=128, kernel_size=5, strides=1, padding='valid', kernel_regularizer=l2(reg_coef)),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(reg_coef)),
        Dense(10, activation='softmax', kernel_regularizer=l2(reg_coef))
    ])

    cnn.compile(
        optimizer=SGD(lr=0.015, momentum=0.9, decay=0.0, nesterov=False),
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    return cnn

def encode_labels(all_labels, labels):
    mapped_labels = np.array(labels)

    mapping = dict()
    for code, label in enumerate(all_labels):
        mapped_labels[mapped_labels == label] = code

    return to_categorical(mapped_labels, len(all_labels))


def process_batch(all_labels, batch):
    features = batch[0]
    labels = batch[1]

    features = (features.astype(np.float) - [123, 117, 104]) * 0.0125
    labels = encode_labels(all_labels, labels)
    return (features, labels)

def conduct(data_dir):
    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    all_labels = list_dirs_in(os.path.join(data_dir, 'train'))
    train, test, _ = train_test_holdout_split(all_samples)

    transforms_and_weights = default_transforms_and_weights()

    validation_data = list(map(partial(process_batch, all_labels), in_batches(None, tqdm(image_generator(
        test,
        partial(random_transform, transforms_and_weights),
        partial(center_crop, 64),
        loop=False
    )))))[0]

    cnn = build_cnn()

    train_generator = map(partial(process_batch, all_labels), in_batches(128, image_generator(
        train,
        partial(random_transform, transforms_and_weights),
        partial(sequential_crop, 64, 15)
    )))

    n_epochs = 50
    n_batches = 10
    for epoch in tqdm(range(n_epochs)):
        learning_rate = learning_schedule(epoch)
        K.set_value(cnn.optimizer.lr, learning_rate)

        for _ in tqdm(range(n_batches)):
            features, labels = next(train_generator)
            cnn.train_on_batch(features, labels)

        train_metrics = cnn.test_on_batch(features, labels)
        test_metrics = cnn.test_on_batch(validation_data[0], validation_data[1])
        tqdm.write('Training ' + str(list(zip(cnn.metrics_names, train_metrics))) + ' Validation ' + str(list(zip(cnn.metrics_names, test_metrics))))
