import os
from functools import partial
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from camera.generators import image_generator, in_batches
from camera.data import list_all_samples_in, train_test_holdout_split
from camera.transforms import random_transform, center_crop, sequential_crop, default_transforms_and_weights, identity, flip_horizontally

def process_image(quantization, threshold, img):
    image, label = img[0], img[1]
    # # 1. Convert to single channel
    # grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # grayscale_image = grayscale_image.astype(np.float)
    grayscale_image = image[:, :, 1].astype(np.float)

    # 2. Calculate residuals
    # spam11 filter 3x3
    filter = np.array([
        [-1, +2, -1],
        [+2, -4, +2],
        [-1, +2, -1]
    ])

    # spam11 filter 5x5
    filter = np.array([
        [-1, +2, -2, +2, -1],
        [+2, -6, +8, -6, +2],
        [-2, +8, -12, +8, -2],
        [+2, -6, +8, -6, +2],
        [-1, +2, -2, +2, -1]
    ])

    # spam14v
    # filter = np.array([
    #     [-1, +2, -1],
    #     [+2, -4, +2],
    #     [+0, +0, +0]
    # ])

    diff = cv2.filter2D(grayscale_image, -1, filter).astype(np.int)
    diff //= quantization
    diff = diff.astype(np.int)
    np.clip(diff, a_min=-threshold, a_max=threshold, out=diff)
    diff += threshold

    # 3. Calculate co-occurrence matricies on a single channel
    dim = 2 * threshold + 1
    patterns = np.zeros((dim, dim, dim, dim), dtype=np.int)
    np.add.at(patterns, [diff[:, ::4], diff[:, 1::4], diff[:, 2::4], diff[:, 3::4]], 1)
    np.add.at(patterns, [diff[::4, :], diff[1::4, :], diff[2::4, :],  diff[3::4, :]], 1)
    patterns = patterns.reshape(-1)
    patterns = patterns / (len(diff[::4, :]) * 2)
    return np.concatenate([patterns[:150], [label]])

def conduct(data_dir):
    quantization = 12
    threshold = 2
    crop_size = 512

    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    train, test, _ = train_test_holdout_split(all_samples)

    transforms_and_weights = default_transforms_and_weights()

    validation_data = np.stack(list(map(partial(process_image, quantization, threshold), tqdm(image_generator(
        test[:100],
        identity, #partial(random_transform, transforms_and_weights),
        partial(center_crop, crop_size),
        loop=False
    )))), axis=0)

    X_test = validation_data[:, :-1]
    y_test = validation_data[:, -1]

    training_data = np.stack(list(map(partial(process_image, quantization, threshold), tqdm(image_generator(
        train[:1000],
        identity, #partial(random_transform, transforms_and_weights),
        partial(center_crop, crop_size),
        loop=False
    )))), axis=0)

    X_train = training_data[:, :-1]
    y_train = training_data[:, -1]

    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
