import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras_contrib.applications import ResidualOfResidual, WideResidualNetwork, DenseNet

from keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    MobileNet,
    InceptionResNetV2,
    InceptionV3,
    ResNet50,
    Xception
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

def residual_of_residual(input_shape, num_classes):
    return Sequential([
        freeze_all_layers(ResidualOfResidual(include_top=False, input_shape=input_shape)),
        Flatten(),
        Dense(num_classes, activation='softmax')
    ])

def wide_residual_network(input_shape, num_classes):
    return Sequential([
        freeze_all_layers(WideResidualNetwork(include_top=False, input_shape=input_shape)),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def mobile_net(input_shape, num_classes):
    return Sequential([
        freeze_all_layers(MobileNet(include_top=False, input_shape=input_shape)),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def inception_resnet_v2(input_shape, num_classes):
     return Sequential([
        freeze_all_layers(InceptionResNetV2(include_top=False, input_shape=input_shape)),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def inception_v3(input_shape, num_classes):
    return Sequential([
        freeze_all_layers(InceptionV3(include_top=False, input_shape=input_shape)),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def resnet_50(input_shape, num_classes):
    return Sequential([
        freeze_all_layers(ResNet50(include_top=False, input_shape=input_shape)),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])

def xception(input_shape, num_classes):
    return Sequential([
        freeze_all_layers(Xception(include_top=False, input_shape=input_shape)),
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])
