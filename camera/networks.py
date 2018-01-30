import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras_contrib.applications import (
    ResidualOfResidual,
    # Shape - 32,32
    # CIFAR10
    # if include_top:
    # x = Flatten()(x)
    # x = Dense(nb_classes, activation='softmax')(x)
    WideResidualNetwork,
    # CIFAR10
    # Shape - 32,32
    # if include_top:
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(nb_classes, activation=activation)(x),
    DenseNet
)
from keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    # Shape - 224, 224
    # IMAGENET
    # if include_top:
    # x = GlobalAveragePooling2D(name='avg_pool')(x)
    # x = Dense(classes, activation='softmax', name='fc1000')(x)
    MobileNet,
    # Shape - 224, 224
    # IMAGENET
    # if include_top:
    #     if K.image_data_format() == 'channels_first':
    #         shape = (int(1024 * alpha), 1, 1)
    #     else:
    #         shape = (1, 1, int(1024 * alpha))

    #     x = GlobalAveragePooling2D()(x)
    #     x = Reshape(shape, name='reshape_1')(x)
    #     x = Dropout(dropout, name='dropout')(x)
    #     x = Conv2D(classes, (1, 1),
    #                padding='same', name='conv_preds')(x)
    #     x = Activation('softmax', name='act_softmax')(x)
    #     x = Reshape((classes,), name='reshape_2')(x)
    InceptionResNetV2,
    # Shape - 299, 299
    # IMAGENET
    # if include_top:
    #     # Classification block
    #     x = GlobalAveragePooling2D(name='avg_pool')(x)
    #     x = Dense(classes, activation='softmax', name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)
    InceptionV3,
    # Shape - 299, 299
    # IMAGENET
    # if include_top:
    #     # Classification block
    #     x = GlobalAveragePooling2D(name='avg_pool')(x)
    #     x = Dense(classes, activation='softmax', name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)
    ResNet50,
    # Shape - 224, 224
    # IMAGENET
    # if include_top:
    #     x = Flatten()(x)
    #     x = Dense(classes, activation='softmax', name='fc1000')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)
    Xception
    # Shape - 299, 299
    # IMAGENET
    # if include_top:
    #     x = GlobalAveragePooling2D(name='avg_pool')(x)
    #     x = Dense(classes, activation='softmax', name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)
)

from keras import backend as K

def drop_layers(n, model):
    for _ in range(n):
        model.layers.pop()

    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]

    return model

def freeze_all_layers(model):
    for layer in model.layers:
        layer.trainable = False

    return model

def unfreeze_layers(n, model):
    unfrozen = 0
    for layer in model.layers[::-1]:
        if not layer.trainable:
            layer.trainable = True
            unfrozen += 1

            if unfrozen == n:
                break

    return model

def densenet_40(input_shape, num_classes):
    return Sequential([
        DenseNet(
            depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=0.0,
            input_shape=input_shape, include_top=False, weights=None),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def densenet_121(input_shape, num_classes):
    return Sequential([
        freeze_all_layers(DenseNet121(include_top=False, input_shape=input_shape)),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def densenet_169(input_shape, num_classes):
    return Sequential([
        freeze_all_layers(DenseNet169(include_top=False, input_shape=input_shape)),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def densenet_201(input_shape, num_classes):
    return Sequential([
        freeze_all_layers(DenseNet169(include_top=False, input_shape=input_shape)),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def learning_rate_schedule(initial_rate, epoch, model):
    lr = initial_rate / (2 ** (epoch // 10))
    K.set_value(model.optimizer.lr, lr)

def get_learning_rate(model):
    return K.get_value(model.optimizer.lr)
