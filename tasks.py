from invoke import task
from camera.shared.data import get_all_labels, load_images, get_image_paths
from camera.feature_extraction.occurrence_matricies import occurrence_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from skimage import io

def process_patch(image, i, j, size, threshold):
    patch = image[i * size:(i + 1) * size, j * size:(j + 1) * size]
    residual_dimension = 2 * threshold + 1
    patterns = np.zeros((residual_dimension, residual_dimension, residual_dimension))
    # TODO AS: Performance eater
    np.add.at(patterns, [patch[::2, ::2], patch[::2, 1::2], patch[1::2, 1::2]], 1)
    patterns /= (patch.shape[0] * patch.shape[1] / 4)
    return patterns.reshape(-1)

@task
def extr(ctx):
    image_id = 0
    size = 512
    threshold = 3
    quantization = 2
    delayed_process_patch = delayed(process_patch)

    with Parallel(n_jobs=-1, backend='threading') as parallel:
        for label in tqdm(get_all_labels()):
            for image_path in tqdm(get_image_paths(label)):
                image = io.imread(image_path)
                try:
                    if (image.shape == ()):
                        # TODO AS: Some images have second thumbnail frame
                        continue

                    # 1. Interpolate
                    red_image = image[:, :, 0].astype(np.float)
                    interpolated_image = np.array(red_image)
                    true_red = red_image[1::2, 0::2]
                    interpolated_image[0::2, 0::2] = true_red
                    interpolated_image[1::2, 1::2] = true_red
                    interpolated_image[0::2, 1::2] = true_red

                    # 2. Calculate difference, quantize and clip
                    diff = red_image - interpolated_image
                    diff /= quantization
                    diff = np.floor(diff).astype(np.int)
                    # TODO AS: Performance eater
                    np.clip(diff, a_min=-threshold, a_max=threshold, out=diff)
                    diff += threshold

                    # 3. For each patch, calculate co-occurrence matrix
                    processed_patches = parallel([delayed_process_patch(diff, i, j, size, threshold) for i in range(diff.shape[0] // size) for j in range(diff.shape[1] // size)])

                    # 4. Append information to the csv
                    df = pd.DataFrame(processed_patches)
                    df['image_id'] = image_id
                    df['label'] = label
                    df.to_csv('./extracted_features.csv', mode='a', header=False)
                except Exception as err:
                    tqdm.write('Error on image {} with label {}: {}'.format(image_id, label, err))
                image_id += 1
