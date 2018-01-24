import os
from functools import partial
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from camera.generators import image_generator, in_batches
from camera.data import list_all_samples_in, train_test_holdout_split
from camera.transforms import random_transform, center_crop, sequential_crop, default_transforms_and_weights, identity

def process_image(quantization, threshold, img):
    image, label = img[0], img[1]
    # 1. Interpolate
    red_image = image[:, :, 0].astype(np.int16)
    interpolated_image = np.array(red_image)
    true_red = red_image[1::2, 0::2]
    interpolated_image[0::2, 0::2] = true_red
    interpolated_image[1::2, 1::2] = true_red
    interpolated_image[0::2, 1::2] = true_red

    # 2. Calculate difference, quantize and clip
    diff = red_image - interpolated_image
    diff //= quantization
    # TODO AS: Performance eater
    np.clip(diff, a_min=-threshold, a_max=threshold, out=diff)
    diff += threshold

    # 3. Calculate co-occurrence matrix
    residual_dimension = 2 * threshold + 1
    patterns = np.zeros((residual_dimension, residual_dimension, residual_dimension), dtype=np.int)
    np.add.at(patterns, [diff[::2, ::2], diff[::2, 1::2], diff[1::2, 1::2]], 1)
    patterns = patterns / (diff.shape[0] * diff.shape[1] / 4)
    return np.concatenate([patterns.reshape(-1), [label]])

def conduct(data_dir):
    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    train, test, _ = train_test_holdout_split(all_samples)

    transforms_and_weights = default_transforms_and_weights()

    validation_data = np.stack(list(map(partial(process_image, 2, 3), tqdm(image_generator(
        test,
        identity, #partial(random_transform, transforms_and_weights),
        partial(center_crop, 512),
        loop=False
    )))), axis=0)

    X_test = validation_data[:, :-1]
    y_test = validation_data[:, -1]

    training_data = map(partial(process_image, 2, 3), tqdm(image_generator(
        train,
        identity, #partial(random_transform, transforms_and_weights),
        partial(sequential_crop, 512, 30),
        loop=False
    )))

    X = list()
    y = list()

    i = 0
    for record in training_data:
        X.append(record[:-1])
        y.append(record[-1])
        i += 1

        if i % 10000 == 0:
            model = RandomForestClassifier(n_estimators=50)
            model.fit(X, y)
            predictions = model.predict(X_test)
            tqdm.write(classification_report(y_test, predictions))

