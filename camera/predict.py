"""Model evaluation and metrics commands"""
import os
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
from dotenv import find_dotenv
from dotenv import load_dotenv
from fire import Fire
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from scipy.stats.mstats import gmean

from camera.data import get_datasets
from camera.networks import load
from camera.pipelines import tta_pipeline
from camera.utils import generate_samples

load_dotenv(find_dotenv())

def predict(
        path, data_dir=os.environ['DATA_DIR'], crop_size=224,
        image_filter=None
    ):

    # TODO AS: Parametrize dataset selection
    validation, _, _ = get_datasets(data_dir)
    labels = np.array(validation)[:, 1].astype(np.int)
    model = load(path)

    pool = ThreadPool(initializer=np.random.seed)
    process_validation_image = partial(tta_pipeline, image_filter, False, crop_size)
    validation_generator = generate_samples(pool, False, process_validation_image, validation)

    predictions = []

    for crops, _, _ in tqdm(validation_generator, total=len(validation)):
        tta_predictions = model.predict_on_batch(np.array(crops))
        predictions.append(gmean(tta_predictions, axis=0))

    predictions = np.array(predictions)

    print(classification_report(labels, np.argmax(predictions, axis=1), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    print(log_loss(labels, predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    print(accuracy_score(labels, np.argmax(predictions, axis=1)))
    print(confusion_matrix(labels, np.argmax(predictions, axis=1), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

if __name__ == '__main__':
    Fire(predict)
