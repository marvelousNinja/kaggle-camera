from invoke import task
from camera.shared.data import get_all_labels, load_images
from camera.feature_extraction.occurrence_matricies import occurrence_matrix
from skimage.util import view_as_blocks
from tqdm import tqdm
import pandas as pd
import numpy as np
from skimage import io
from matplotlib import pyplot as plt

@task
def extr(ctx):
    sample_id = 0
    image_id = 0
    patch_size = 512

    for label in tqdm(get_all_labels()):
        for image in tqdm(load_images(label)[:20]):
            try:
                image_patches = list()
                if (image.shape == ()):
                    # TODO AS: Some images have second thumbnail frame
                    continue
                for i in range(image.shape[0] // patch_size)[:4]:
                    for j in range(image.shape[1] // patch_size)[:4]:
                        patch = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                        # TODO AS: Optimal threshold value
                        features = np.append(occurrence_matrix(patch, threshold=2).reshape(-1), [image_id, label])
                        image_patches.append(features)
                        sample_id += 1
                df = pd.DataFrame(image_patches)
                df.to_csv('./extracted_features.csv', mode='a', header=False)
            except Exception as err:
                tqdm.write('Error on image {} with label {}: {}'.format(image_id, label, err))

            image_id += 1
