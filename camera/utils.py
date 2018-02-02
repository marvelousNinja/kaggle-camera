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

def encode_labels(all_labels, labels):
    mapped_labels = np.array(labels)
    for code, label in enumerate(all_labels):
        mapped_labels[mapped_labels == label] = code
    return mapped_labels

def evolve_at(key, func, target):
    target[key] = func(target[key])
    return target

def transform_to_sample_weight(transform_name):
    if transform_name == 'identity':
        return 0.7
    else:
        return 0.3
