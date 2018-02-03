import os
from functools import partial
from multiprocessing.pool import ThreadPool
from fire import Fire
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
from camera.utils import in_batches, generate_samples, generate_submission_name
from camera.pipelines import submission_pipeline
from camera.data import get_test_dataset, inverse_label_mapping
from camera.networks import load

load_dotenv(find_dotenv())

def make_submission(
        path, data_dir=os.environ['DATA_DIR'], batch_size=16, crop_size=224,
        image_filter='spam_11_5'
    ):
    test = get_test_dataset(data_dir)[:batch_size]

    pool = ThreadPool(initializer=np.random.seed)
    process_submission_image = partial(submission_pipeline, image_filter, crop_size)
    submission_generator = generate_samples(pool, False, process_submission_image, test)
    submission_generator = in_batches(batch_size, submission_generator)

    model = load(path)
    predictions = model.predict_generator(
        generator=submission_generator,
        steps=int(np.ceil(len(test) / batch_size)),
        verbose=1
    )

    network_name = os.path.basename(path)
    submission_path = os.path.join(data_dir, 'submissions', generate_submission_name(network_name))
    label_mapping = inverse_label_mapping(os.path.join(data_dir, 'train'))
    df = pd.DataFrame({
        'fname': list(map(lambda path: os.path.basename(path), test)),
        'camera': list(map(lambda code: label_mapping[code], np.argmax(predictions, axis=1)))
    })

    df.to_csv(submission_path, index=False)
    print(f'Submission file generated at {submission_path}')

if __name__ == '__main__':
    Fire(make_submission)
