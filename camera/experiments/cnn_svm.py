import os
from multiprocessing import Pool, Manager
from queue import Empty
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
from camera.data import list_dirs_in
from camera.data import list_images_in
from camera.transforms import adjust_gamma
from camera.transforms import flip_horizontally
from camera.transforms import jpeg_compress
from camera.transforms import resize
from camera.transforms import identity

def list_all_samples_in(dir):
    image_paths_and_labels = []

    for label in list_dirs_in(dir):
        image_paths = list_images_in(os.path.join(dir, label))
        labels = [label] * len(image_paths)
        image_paths_and_labels.extend(zip(image_paths, labels))

    return image_paths_and_labels

def train_test_holdout_split(samples, seed=11):
    np.random.seed(seed)
    shuffled_samples = np.array(samples)
    np.random.shuffle(shuffled_samples)
    sample_count = len(samples)
    train = shuffled_samples[0:int(sample_count * 0.7)]
    test = shuffled_samples[int(sample_count * 0.7):int(sample_count * 0.85)]
    holdout = shuffled_samples[int(sample_count * 0.85):]
    return train, test, holdout

def random_transform(transforms_and_weights, image):
    transforms, weights = zip(*transforms_and_weights)
    weights = np.array(weights)
    probabilities = weights / sum(weights)
    transfrom = np.random.choice(transforms, p=probabilities)
    return transfrom(image)

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

def sequential_crop(crop_size, limit, image):
    threshold = 0.5
    patches = []
    for i in range(image.shape[0] // crop_size):
        for j in range(image.shape[1] // crop_size):
            patches.append(image[i * crop_size:(i + 1) * crop_size, j * crop_size:(j + 1) * crop_size])

    patch_scores = []
    for patch in patches:
        patch_scores.append(intensity_texture_score(patch))

    scored_score_indicies = np.argsort(patch_scores)[::-1]
    sorted_scores = np.array(patch_scores)[scored_score_indicies]
    sorted_patches = np.array(patches)[scored_score_indicies]
    return sorted_patches[sorted_scores >= threshold][:limit]

def center_crop(size, image):
    center_x = image.shape[0] // 2 - 1
    center_y = image.shape[1] // 2 - 1
    top_x, top_y = center_x - size // 2, center_y - size // 2
    return [image[top_x:top_x + size, top_y:top_y + size]]

def process_image(path, label, transform, crop, queue):
    try:
        image = cv2.imread(path)
        transformed_image = transform(image)
        rgb_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB).astype(np.int64)
        for patch in crop(rgb_image):
            queue.put((True, (patch, label)))
    except Exception as e:
        queue.put((False, e))

def image_generator(image_paths_and_labels, transform, crop, loop=True, seed=11):
    np.random.seed(seed)

    while True:
        try:
            pool = Pool(processes=2, initializer=np.random.seed)
            manager = Manager()
            image_patch_queue = manager.Queue(200)

            pairs = np.array(image_paths_and_labels)
            np.random.shuffle(pairs)

            for image_path, label in pairs:
                pool.apply_async(process_image, (image_path, label, transform, crop, image_patch_queue))

            while True:
                success, response = image_patch_queue.get()
                if success:
                    yield response
                else:
                    raise response

        except Empty:
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

    features -= [123, 117, 104]
    features = features * 0.0125

    labels = encode_labels(all_labels, labels)

    return (features, labels)

def conduct(data_dir):
    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    all_labels = list_dirs_in(os.path.join(data_dir, 'train'))
    train, test, _ = train_test_holdout_split(all_samples)

    transforms_and_weights = (
        (identity, 8),
        (partial(adjust_gamma, 0.8), 1),
        (partial(adjust_gamma, 1.2), 1),
        (partial(jpeg_compress, 70), 1),
        (partial(jpeg_compress, 90), 1),
        (partial(resize, 0.5), 1),
        (partial(resize, 0.8), 1),
        (partial(resize, 1.5), 1),
        (partial(resize, 2.0), 1)
    )

    train_generator = map(partial(process_batch, all_labels), in_batches(128, image_generator(
        train,
        partial(random_transform, transforms_and_weights),
        partial(sequential_crop, 64, 25)
    )))

    validation_data = list(islice(map(partial(process_batch, all_labels), in_batches(128, image_generator(
        test,
        partial(random_transform, transforms_and_weights),
        partial(center_crop, 64)
    ))), 1))[0]

    cnn = build_cnn()

    n_epochs = 50
    n_batches = 200
    for epoch in tqdm(range(n_epochs)):
        learning_rate = learning_schedule(epoch)
        K.set_value(cnn.optimizer.lr, learning_rate)

        for _ in tqdm(range(n_batches)):
            features, labels = next(train_generator)
            cnn.train_on_batch(features, labels)

        train_metrics = cnn.test_on_batch(features, labels)
        test_metrics = cnn.test_on_batch(validation_data[0], validation_data[1])
        tqdm.write('Training ' + str(list(zip(cnn.metrics_names, train_metrics))) + ' Validation ' + str(list(zip(cnn.metrics_names, test_metrics))))
