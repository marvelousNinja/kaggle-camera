from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D, Convolution2D
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2

# TODO AS: LR should half every 10 epochs
def learning_schedule(epoch):
    return 0.015

def conduct():
    reg_coef = 0.00075

    cnn = Sequential([
        Convolution2D(filters=32, kernel_size=4, strides=1, padding='valid', input_shape=(64, 64, 3), kernel_regularizer=l2(reg_coef)),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        Convolution2D(filters=48, kernel_size=5, strides=1, padding='valid', kernel_regularizer=l2(reg_coef)),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        Convolution2D(filters=64, kernel_size=5, strides=1, padding='valid', kernel_regularizer=l2(reg_coef)),
        MaxPooling2D(pool_size=2, strides=2, padding='same'),
        Convolution2D(filters=128, kernel_size=5, strides=1, padding='valid', kernel_regularizer=l2(reg_coef)),
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(reg_coef)),
        Dense(10, activation='softmax', kernel_regularizer=l2(reg_coef))
    ])

    cnn.compile(
        optimizer=SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False),
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    # TODO AS: Train generator in `generator` param
    # TODO AS: Steps per epoch - should be set manually, since generator is expeced to iterate forever
    # TODO AS: Cache validation data?
    # TODO AS: Validation steps - if I don't cache it, how many batches?
    cnn.fit_generator(
        generator=None,
        steps_per_epoch=None,
        epochs=50,
        verbose=2,
        callbacks=[LearningRateScheduler(learning_schedule, verbose=2)],
        validation_data=None,
        validation_steps=None
    )
