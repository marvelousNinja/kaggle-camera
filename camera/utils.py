import os
from datetime import datetime

import numpy as np

def pipe(funcs, target):
    result = target
    for func in funcs:
        result = func(result)
    return result

def in_batches(batch_size, iterable):
    batch = list()
    for element in iterable:
        batch.append(element)
        if len(batch) == batch_size:
            yield np.stack(batch)
            batch = list()
    if len(batch) > 0:
        yield np.stack(batch)

def only_at(key, iterable):
    return map(lambda obj: obj[key], iterable)

def in_x_y_s_batches(batch_size, iterable):
    x_batch = list()
    y_batch = list()
    sample_weights_batch = list()
    for x, y, sample_weight in iterable:
        x_batch.append(x)
        y_batch.append(y)
        sample_weights_batch.append(sample_weight)

        if len(x_batch) == batch_size:
            yield np.stack(x_batch), np.stack(y_batch), np.stack(sample_weights_batch)
            x_batch = list()
            y_batch = list()
            sample_weights_batch = list()

    if len(x_batch) > 0:
        yield np.stack(x_batch), np.stack(y_batch), np.stack(sample_weights_batch)

def evolve_at(key, func, target):
    target[key] = func(target[key])
    return target

def transform_to_sample_weight(transform_name):
    if transform_name == 'identity':
        return 0.7
    else:
        return 0.3

def generate_model_name(network, crop_size):
    timestr = datetime.utcnow().strftime('%Y%m%d_%H%M')
    return f'{network}-{crop_size}-{timestr}-' + '{epoch:02d}-{val_acc:.5f}-{val_loss:.5f}.hdf5'

def generate_submission_name(network):
    timestr = datetime.utcnow().strftime('%Y%m%d_%H%M')
    return f'submission-{timestr}-{network[:-5]}.csv'

def generate_blend_submission_name(files):
    shortened_name = '__'.join(map(lambda path: os.path.basename(path)[11:35], files))
    return f'blend-{shortened_name}.csv'

def generate_samples(pool, shuffle, pipeline, records):
    records = list(records)
    if shuffle: np.random.shuffle(records)
    return pool.imap(pipeline, records)

def in_loop(initializer):
    while True: yield from initializer()
