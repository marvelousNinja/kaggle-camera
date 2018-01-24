import os
from functools import partial
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from camera.generators import image_generator, in_batches
from camera.data import list_all_samples_in, train_test_holdout_split
from camera.transforms import random_transform, center_crop, sequential_crop, default_transforms_and_weights, identity

def process_image(quantization, threshold, img):
    image, label = img[0], img[1]
    # 1. Denoise image
    # bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # denoised_image = cv2.fastNlMeansDenoisingColored(bgr_image, h=3)
    # rgb_image = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)

    rgb_image = denoise_wavelet(image)
    # 2. Calculate residuals
    diff = image.astype(np.int) - rgb_image.astype(np.int)
    diff //= quantization
    np.clip(diff, a_min=-threshold, a_max=threshold, out=diff)
    diff += threshold

    # 3. Calculate co-occurrence matricies on a single channel
    diff = diff[:, :, 1]
    dim = 2 * threshold + 1
    horizontal_patterns = np.zeros((dim, dim, dim, dim), dtype=np.int)
    np.add.at(horizontal_patterns, [diff[:, ::4], diff[:, 1::4], diff[:, 2::4], diff[:, 3::4]], 1)
    vertical_patterns = np.zeros((dim, dim, dim, dim), dtype=np.int)
    np.add.at(vertical_patterns, [diff[::4, :], diff[1::4, :], diff[2::4, :],  diff[3::4, :]], 1)
    return np.concatenate([horizontal_patterns.reshape(-1), vertical_patterns.reshape(-1), [label]])

def conduct(data_dir):
    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    train, test, _ = train_test_holdout_split(all_samples)

    transforms_and_weights = default_transforms_and_weights()

    validation_data = np.stack(list(map(partial(process_image, 2, 2), tqdm(image_generator(
        test,
        identity, #partial(random_transform, transforms_and_weights),
        partial(center_crop, 224),
        loop=False
    )))), axis=0)

    X_test = validation_data[:, :-1]
    y_test = validation_data[:, -1]

    training_data = map(partial(process_image, 5, 2), tqdm(image_generator(
        train,
        identity, #partial(random_transform, transforms_and_weights),
        partial(sequential_crop, 224, 25),
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

