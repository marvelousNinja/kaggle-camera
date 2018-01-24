from functools import partial
import numpy as np
from camera.transforms.adjust_gamma import adjust_gamma
from camera.transforms.resize import resize
from camera.transforms.jpeg_compress import jpeg_compress
from camera.transforms.flip_horizontally import flip_horizontally

def identity(image):
    return image

def default_transforms_and_weights():
    return (
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

def random_transform(transforms_and_weights, image):
    transforms, weights = zip(*transforms_and_weights)
    weights = np.array(weights)
    probabilities = weights / sum(weights)
    transform = np.random.choice(transforms, p=probabilities)
    return transform(image)

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
