import os
from multiprocessing import Pool, cpu_count
from functools import partial
import itertools

from tqdm import tqdm
import numpy as np

from camera.networks import densenet_40
from camera.utils import pipe, encode_labels, in_batches
from camera.data import list_all_samples_in, train_test_holdout_split, list_dirs_in
from camera.data import read_jpeg
from camera.transforms import (
    default_transforms_and_weights, random_transform,
    crop_center, crop_random,
    spam_11_3, spam_11_5, spam_14_edge,
    identity
)

def conduct(
        data_dir,
        crop_size,
        n_epochs,
        batch_size,
        outer_crop_strategy,
        inner_crop_strategy,
        residual_filter_strategy,
        test_limit,
        train_limit
    ):

    outer_crop_size = crop_size * 2 + 8
    input_shape = (crop_size, crop_size, 3)

    all_samples = list_all_samples_in(os.path.join(data_dir, 'train'))
    all_labels = list_dirs_in(os.path.join(data_dir, 'train'))
    train, test, _ = train_test_holdout_split(all_samples)
    test = test[:test_limit]
    train = train[:train_limit]

    n_batches = int(np.ceil(len(train) / batch_size))

    crop_strategies = {
        None: identity,
        'crop_center': crop_center,
        'crop_random': crop_random
    }

    filter_strategies = {
        None: identity,
        'spam_11_3': spam_11_3,
        'spam_11_5': spam_11_5,
        'spam_14_edge': spam_14_edge
    }

    train_pipeline = partial(pipe, [
        read_jpeg,
        partial(crop_strategies[outer_crop_strategy], outer_crop_size),
        partial(random_transform, default_transforms_and_weights()),
        filter_strategies[residual_filter_strategy],
        partial(crop_strategies[inner_crop_strategy], crop_size)
    ])

    test_pipeline = partial(pipe, [
        read_jpeg,
        partial(crop_center, outer_crop_size),
        partial(random_transform, default_transforms_and_weights()),
        filter_strategies[residual_filter_strategy],
        partial(crop_center, crop_size)
    ])

    pool = Pool(processes=cpu_count() - 2, initializer=np.random.seed)

    label_encoder = partial(encode_labels, all_labels)
    to_batch = partial(in_batches, batch_size)

    paths, labels = zip(*test)
    x_test = np.stack(list(tqdm(pool.imap(test_pipeline, paths))))
    y_test = np.stack(label_encoder(labels))

    model = densenet_40()

    for epoch in tqdm(range(n_epochs)):
        np.random.shuffle(train)
        paths, labels = zip(*train)
        labels = label_encoder(labels)

        for x_train, y_train in tqdm(zip(to_batch(pool.imap(train_pipeline, paths)), to_batch(labels)), total=n_batches):
            model.train_on_batch(x_train, y_train)

        train_metrics = model.test_on_batch(x_train, y_train)
        test_metrics = model.test_on_batch(x_test, y_test)
        tqdm.write('Training ' + str(list(zip(model.metrics_names, train_metrics))) + ' Validation ' + str(list(zip(model.metrics_names, test_metrics))))
