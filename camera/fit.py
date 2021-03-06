"""Model training commands"""
import os
from functools import partial
from multiprocessing.pool import ThreadPool

import numpy as np
from dotenv import find_dotenv
from dotenv import load_dotenv
from fire import Fire
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

from camera.callbacks import Unfreeze
from camera.callbacks import SwitchOptimizer
from camera.callbacks import WarmRestartSGD
from camera.data import get_datasets
from camera.networks import get_model
from camera.pipelines import training_pipeline
from camera.pipelines import validation_pipeline
from camera.custom_datasets import get_scrapped_dataset
from camera.utils import calculate_class_weights
from camera.utils import generate_model_name
from camera.utils import generate_samples
from camera.utils import in_loop
from camera.utils import in_x_y_s_batches

load_dotenv(find_dotenv())

def fit(
        data_dir=os.environ['DATA_DIR'], lr=0.0001, batch_size=16,
        crop_size=224, network=None, image_filter=None, overfit_run=False,
        allow_weights=True, allow_flips=True, callbacks=['switch', 'reduce_lr'],
        unfreeze_at=0, switch_at=13
    ): # pylint: disable=too-many-arguments,dangerous-default-value

    # TODO AS: Parametrize the dataset
    _, validation, _ = get_datasets(data_dir)
    train = get_scrapped_dataset()
    np.random.shuffle(train)

    if overfit_run:
        train = train[:batch_size]
        validation = validation[:batch_size]

    pool = ThreadPool(initializer=np.random.seed)
    process_training_image = partial(
        training_pipeline, dict(), image_filter, allow_flips, allow_weights, crop_size
    )

    training_generator = in_loop(lambda: generate_samples(pool, True, process_training_image, train))
    training_generator = in_x_y_s_batches(batch_size, training_generator)

    process_validation_image = partial(
        validation_pipeline, image_filter, allow_weights, crop_size
    )

    validation_data = list(generate_samples(pool, True, process_validation_image, validation))
    validation_generator = in_loop(lambda: validation_data)
    validation_generator = in_x_y_s_batches(batch_size, validation_generator)

    additional_callbacks = {
        'switch': SwitchOptimizer(switch_at, SGD(lr, momentum=0.9, nesterov=True), verbose=1),
        'reduce_lr': ReduceLROnPlateau(patience=4, min_lr=0.1e-6, factor=0.1, verbose=1),
        'sgdr': WarmRestartSGD(switch_at, int(np.ceil(len(train) / batch_size)), max_lr=lr, verbose=1)
    }

    model = get_model(network)((crop_size, crop_size, 3), 10)
    print(model.summary())
    model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy', metrics=['acc'])

    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        steps_per_epoch=int(np.ceil(len(train) / batch_size)),
        validation_steps=int(np.ceil(len(validation) / batch_size)),
        class_weight=calculate_class_weights(train[:, 1]),
        epochs=200,
        verbose=2,
        callbacks=[
            Unfreeze(unfreeze_at, verbose=1),
            ModelCheckpoint(os.path.join(data_dir, 'models', generate_model_name(network, crop_size)), save_best_only=True, verbose=1),
        ] + [additional_callbacks[callback] for callback in callbacks]
    )

if __name__ == '__main__':
    Fire(fit)
