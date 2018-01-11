from invoke import task
from camera.shared.data import get_all_labels, load_images
from camera.feature_extraction.occurrence_matricies import occurrence_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def process_patch(image, i, j, size, threshold):
    patch = image[i * size:(i + 1) * size, j * size:(j + 1) * size]
    return occurrence_matrix(patch, threshold).reshape(-1)

@task
def extr(ctx):
    image_id = 0
    patch_size = 512
    threshold = 2
    delayed_process_patch = delayed(process_patch)

    with Parallel(n_jobs=-1, backend='threading') as parallel:
        for label in tqdm(get_all_labels()):
            for image in tqdm(load_images(label)[:20]):
                try:
                    if (image.shape == ()):
                        # TODO AS: Some images have second thumbnail frame
                        continue
                    for i in range(image.shape[0] // patch_size):
                       image_patches = parallel(delayed_process_patch(image, i, j, patch_size, threshold) for j in range(image.shape[1] // patch_size))
                    df = pd.DataFrame(image_patches)
                    df['image_id'] = image_id
                    df['label'] = label
                    df.to_csv('./extracted_features.csv', mode='a', header=False)
                except Exception as err:
                    tqdm.write('Error on image {} with label {}: {}'.format(image_id, label, err))
                image_id += 1
