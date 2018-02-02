from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

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

def drop_layers(n, model):
    for _ in range(n):
        model.layers.pop()

    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]

    return model

def freeze_all_layers(model):
    for layer in model.layers:
        layer.trainable = False
        if hasattr(layer, 'layers'):
            freeze_all_layers(layer)

    return model

def unfreeze_all_layers(model):
    for layer in model.layers:
        layer.trainable = True
        if hasattr(layer, 'layers'):
            unfreeze_all_layers(layer)

    return model

def densenet_121(input_shape, num_classes):
    model = freeze_all_layers(DenseNet121(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)

def densenet_169(input_shape, num_classes):
    model = freeze_all_layers(DenseNet169(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)

def densenet_201(input_shape, num_classes):
    model = freeze_all_layers(DenseNet201(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)

def mobile_net(input_shape, num_classes):
    model = freeze_all_layers(MobileNet(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)

def inception_resnet_v2(input_shape, num_classes):
    model = freeze_all_layers(InceptionResNetV2(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)


def inception_v3(input_shape, num_classes):
    model = freeze_all_layers(InceptionV3(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)


def resnet_50(input_shape, num_classes):
    model = freeze_all_layers(ResNet50(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)

def xception(input_shape, num_classes):
    model = freeze_all_layers(Xception(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)
