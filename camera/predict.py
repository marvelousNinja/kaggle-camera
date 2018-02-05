import os
from functools import partial
from multiprocessing.pool import ThreadPool
from fire import Fire
from dotenv import load_dotenv, find_dotenv
import numpy as np
from camera.utils import generate_model_name, in_x_y_s_batches, generate_samples, only_at
from camera.pipelines import validation_pipeline
from camera.data import get_datasets, get_flickr_dataset, get_reviews_dataset
from camera.networks import load
from sklearn.metrics import classification_report, log_loss, confusion_matrix, accuracy_score

load_dotenv(find_dotenv())

def predict(
        path, data_dir=os.environ['DATA_DIR'], batch_size=16, crop_size=224,
        image_filter='spam_11_5'
    ):

    holdout = get_flickr_dataset(data_dir)

    holdout = holdout
    labels = np.array(holdout)[:, 1].astype(np.int)

    model = load(path)

    pool = ThreadPool(initializer=np.random.seed)
    process_validation_image = partial(validation_pipeline, image_filter, False, crop_size)
    validation_generator = generate_samples(pool, False, process_validation_image, holdout)
    validation_generator = in_x_y_s_batches(batch_size, validation_generator)
    validation_generator = only_at(0, validation_generator)

    predictions = model.predict_generator(
        generator=validation_generator,
        steps=int(np.ceil(len(holdout) / batch_size)),
        verbose=1
    )

    print(classification_report(labels, np.argmax(predictions, axis=1), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    print(log_loss(labels, predictions, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    print(accuracy_score(labels, np.argmax(predictions, axis=1)))
    print(confusion_matrix(labels, np.argmax(predictions, axis=1), labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

if __name__ == '__main__':
    Fire(predict)
