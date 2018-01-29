from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling2D, Convolution2D, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2
from keras_contrib.applications import DenseNet
from keras import backend as K

def drop_n_and_freeze(n, model):
    for _ in range(n):
        model.layers.pop()

    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]

    for layer in model.layers:
        layer.trainable = False

    return model

def densenet_40(input_shape, lr, weights):
    model = DenseNet(
        depth=40,
        nb_dense_block=3,
        growth_rate=12,
        nb_filter=16,
        dropout_rate=0.0,
        input_shape=input_shape,
        pooling='avg',
        include_top=False,
        weights=None
    )

    model.compile(
        optimizer=Adam(lr),
        loss='categorical_crossentropy',
        metrics=['acc']
    )

    return model
