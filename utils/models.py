"""
CNN models.

File implementing the models to use for training and testing.
"""

from keras import models
from keras import layers
from keras import utils
from contextlib import redirect_stdout


def CNN3D(input_shape=(203, 197, 189, 3), weights=False, summary=True):
    """
    Define base 3D CNN implementation.

    Implement a 3D CNN for two-way classification following the architecture
    of Basu et al.
    """
    img_input = layers.Input(shape=input_shape)
    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='conv1',
                      strides=(1, 1, 1), padding="valid")(img_input)
    x = layers.BatchNormalization(axis=4, name='bn1')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool1')(x)

    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='conv2', strides=(1, 1, 1), padding="valid")(x)
    x = layers.BatchNormalization(axis=4, name='bn2')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool2')(x)

    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='conv3',
                      strides=(1, 1, 1), padding="valid")(x)
    x = layers.BatchNormalization(axis=4, name='bn3')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool3')(x)

    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='conv4',
                      strides=(1, 1, 1), padding="valid")(x)
    x = layers.BatchNormalization(axis=4, name='bn4')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool4')(x)

    x = layers.Flatten()(x)

    x = layers.Dense(4096, name='fc0', activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(4096, name='fc1', activation='relu')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(2, activation='softmax', name='fc3')(x)
    model = models.Model(img_input, x)

    with open('modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            utils.print_summary(model, line_length=110, print_fn=None)

    return model


def AE3D(input_shape):
    """
    Base 3D Autoencoder implementation.

    Implement a 3D Autoencoder for two-way classification following the architecture of Basu et al.
    """
    img_input = layers.Input(shape=input_shape)

    # Decoder
    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='d_conv1',
                      strides=(1, 1, 1), padding="valid")(img_input)
    x = layers.BatchNormalization(axis=4, name='d_bn1')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='d_pool1')(x)

    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='d_conv2', strides=(1, 1, 1), padding="valid")(x)
    x = layers.BatchNormalization(axis=4, name='d_bn2')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='d_pool2')(x)

    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='conv3',
                      strides=(1, 1, 1), padding="valid")(x)
    x = layers.BatchNormalization(axis=4, name='d_bn3')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='d_pool3')(x)

    x = layers.Conv3D(11, (3, 3, 3), activation='relu', name='d_conv4',
                      strides=(1, 1, 1), padding="valid")(x)
    x = layers.BatchNormalization(axis=4, name='d_bn4')(x)
    encoded = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='d_pool4')(x)

    ## Encoder
    x = layers.UpSampling3D(size=(2, 2, 2), name='e_upsampl1')(x)
    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='e_conv1',
                      strides=(1, 1, 1), padding="valid")(x)
    x = layers.BatchNormalization(axis=4, name='e_bn1')(x)

    x = layers.UpSampling3D(size=(2, 2, 2), name='e_upsampl2')(x)
    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='e_conv2',
                      strides=(1, 1, 1), padding="valid")(x)
    x = layers.BatchNormalization(axis=4, name='e_bn2')(x)

    x = layers.UpSampling3D(size=(2, 2, 2), name='e_upsampl3')(x)
    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='e_conv3',
                      strides=(1, 1, 1), padding="valid")(x)
    x = layers.BatchNormalization(axis=4, name='e_bn3')(x)

    x = layers.UpSampling3D(size=(2, 2, 2), name='e_upsampl4')(x)
    x = layers.Conv3D(11, (3, 3, 3), activation='relu',
                      name='e_conv4',
                      strides=(1, 1, 1), padding="valid")(x)
    decoded = layers.BatchNormalization(axis=4, name='e_bn4')(x)

    ## Normal output
    x = layers.Flatten()(encoded)

    x = layers.Dense(4096, name='fc0', activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(4096, name='fc1', activation='relu')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    output_base = layers.Dense(2, activation='softmax', name='fc3')(x)
    model = models.Model(inputs=[img_input], outputs=[output_base, decoded])
    return model
