from keras.models import Model, load_model
from keras.layers import Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.generic_utils import CustomObjectScope

from keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    MobileNet,
    InceptionResNetV2,
    InceptionV3,
    ResNet50,
    Xception,
    NASNetMobile,
    NASNetLarge
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
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
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

def nasnet_mobile(input_shape, num_classes):
    model = freeze_all_layers(NASNetMobile(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)

def nasnet_large(input_shape, num_classes):
    model = freeze_all_layers(NASNetLarge(include_top=False, input_shape=input_shape))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=model.input, outputs=x)

def get_model(name):
    return {
        'densenet_121': densenet_121,
        'densenet_169': densenet_169,
        'densenet_201': densenet_201,
        'mobile_net': mobile_net,
        'inception_resnet_v2': inception_resnet_v2,
        'inception_v3': inception_v3,
        'resnet_50': resnet_50,
        'xception': xception,
        'nasnet_mobile': nasnet_mobile,
        'nasnet_large': nasnet_large
    }[name]

def load(path):
    with CustomObjectScope({ 'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D }):
        return load_model(path)
