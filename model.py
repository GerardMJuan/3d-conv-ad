"""
Implementation of the neural network.

It is based on Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren
T. (2017) On the Compactness, Efficiency, and Representation of 3D
Convolutional Networks: Brain Parcellation as a Pretext Task. In:
Niethammer M. et al. (eds) Information Processing in Medical Imaging.
IPMI 2017. Lecture Notes in Computer Science, vol 10265. Springer, Cham.
DOI: 10.1007/978-3-319-59050-9_28
"""

from keras.layers.convolutional import (Conv3D, MaxPooling3D,
                                        ZeroPadding3D)
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, Model


def c3d_base(weights=False, summary=True, weights_file=None):
    """
    Build base model for c3D using Seqential API.

    This model builds a base c3D network.
    """
    c3d_model = Sequential()
    # 1st layer group
    c3d_model.add(Conv3D(64, (3, 3, 3), activation='relu',
                         name='conv1',
                         input_shape=(16, 112, 112, 3),
                         strides=(1, 1, 1), padding="same"))

    c3d_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               padding='valid', name='pool1'))

    # 2nd layer group
    c3d_model.add(Conv3D(128, (3, 3, 3), activation='relu',
                         padding='same', name='conv2',
                         strides=(1, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               padding='valid', name='pool2'))

    # 3rd layer group
    c3d_model.add(Conv3D(256, (3, 3, 3), activation='relu',
                         padding='same', name='conv3a',
                         strides=(1, 1, 1)))
    c3d_model.add(Conv3D(256, (3, 3, 3), activation='relu',
                         padding='same', name='conv3b',
                         strides=(1, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               padding='valid', name='pool3'))

    # 4th layer group
    c3d_model.add(Conv3D(512, (3, 3, 3), activation='relu',
                         padding='same', name='conv4a',
                         strides=(1, 1, 1)))
    c3d_model.add(Conv3D(512, (3, 3, 3), activation='relu',
                         padding='same', name='conv4b',
                         strides=(1, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               padding='valid', name='pool4'))

    # 5th layer group
    c3d_model.add(Conv3D(512, (3, 3, 3), activation='relu',
                         padding='same', name='conv5a',
                         strides=(1, 1, 1)))
    c3d_model.add(Conv3D(512, (3, 3, 3), activation='relu',
                         padding='same', name='conv5b',
                         strides=(1, 1, 1)))

    c3d_model.add(ZeroPadding3D(padding=(0, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               padding='valid', name='pool5'))
    c3d_model.add(Flatten())

    # FC layers group
    c3d_model.add(Dense(4096, activation='relu', name='fc6'))
    c3d_model.add(Dropout(.5))
    c3d_model.add(Dense(4096, activation='relu', name='fc7'))
    c3d_model.add(Dropout(.5))
    c3d_model.add(Dense(487, activation='softmax', name='fc8'))

    if weights:
        c3d_model.load_weights(weights_file)

    # Create new model
    # Get input
    new_input = c3d_model.input
    # Find the layer to connect
    hidden_layer = c3d_model.layers[-2].output
    # Connect a new layer on it
    new_output = Dense(2, activation='softmax', name='fc9')(hidden_layer)
    # Build a new model
    c3d_model_2 = Model(new_input, new_output)

    # Remove last layer and add our own

    # Fix all layers but last 3
    for layer in c3d_model_2.layers[:15]:
        layer.trainable = False

    if summary:
        c3d_model_2.summary()
    return c3d_model_2
