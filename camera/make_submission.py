import os
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from fire import Fire
from scipy.stats.mstats import gmean
from tqdm import tqdm

from camera.data import get_test_dataset
from camera.data import inverse_label_mapping
from camera.networks import load
from camera.pipelines import tta_submission_pipeline
from camera.utils import generate_samples
from camera.utils import generate_submission_name
from camera.utils import in_batches

load_dotenv(find_dotenv())

def make_submission(
        path, data_dir=os.environ['DATA_DIR'], batch_size=16, crop_size=224,
        image_filter=None
    ):
    test = get_test_dataset(data_dir)

    pool = ThreadPool(initializer=np.random.seed)
    process_submission_image = partial(tta_submission_pipeline, image_filter, crop_size)
    submission_generator = generate_samples(pool, False, process_submission_image, test)

    predictions = []

    model = load(path)
    for crops in tqdm(submission_generator, total=len(test)):
        tta_predictions = model.predict_on_batch(np.array(crops))
        predictions.append(gmean(tta_predictions, axis=0))

    predictions = np.array(predictions)

    network_name = os.path.basename(path)
    submission_path = os.path.join(data_dir, 'submissions', generate_submission_name(network_name))
    label_mapping = inverse_label_mapping()
    df = pd.DataFrame({
        'fname': list(map(lambda path: os.path.basename(path), test)),
        'camera': list(map(lambda code: label_mapping[code], np.argmax(predictions, axis=1)))
    })

    df.to_csv(submission_path, index=False)
    print(f'Submission file generated at {submission_path}')

if __name__ == '__main__':
    Fire(make_submission)
