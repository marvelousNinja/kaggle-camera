import os
from functools import partial
from multiprocessing.pool import ThreadPool
from fire import Fire
from dotenv import load_dotenv, find_dotenv
import numpy as np
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from camera.callbacks import Unfreeze, SwitchOptimizer, WarmRestartSGD
from camera.networks import get_model
from camera.utils import generate_model_name, in_x_y_s_batches, in_loop, generate_samples
from camera.pipelines import training_pipeline, validation_pipeline
from camera.data import get_datasets

load_dotenv(find_dotenv())

def fit(
        data_dir=os.environ['DATA_DIR'], lr=0.0001, batch_size=16,
        crop_size=224, network='mobile_net', image_filter='spam_11_5', overfit_run=False,
        allow_weights=True, allow_flips=True, callbacks=['switch', 'reduce_lr']
    ):

    train, validation, _ = get_datasets(data_dir)
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
        'switch': SwitchOptimizer(10, SGD(lr, momentum=0.9, nesterov=True), verbose=1),
        'reduce_lr': ReduceLROnPlateau(patience=10, min_lr=0.5e-6, factor=np.sqrt(0.1), verbose=1),
        'sgdr': WarmRestartSGD(10, int(np.ceil(len(train) / batch_size)), verbose=1)
    }

    model = get_model(network)((crop_size, crop_size, 3), 10)
    print(model.summary())
    model.compile(optimizer=Adam(lr), loss='sparse_categorical_crossentropy', metrics=['acc'])

    model.fit_generator(
        generator=training_generator,
        validation_data=validation_generator,
        steps_per_epoch=int(np.ceil(len(train) / batch_size)),
        validation_steps=int(np.ceil(len(validation) / batch_size)),
        epochs=200,
        verbose=2,
        callbacks=[
            Unfreeze(0, verbose=1),
            ModelCheckpoint(os.path.join(data_dir, 'models', generate_model_name(network, crop_size)), save_best_only=True, verbose=1),
        ] + [additional_callbacks[callback] for callback in callbacks]
    )

if __name__ == '__main__':
    Fire(fit)
