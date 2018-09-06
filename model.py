"""
Implementation of the neural network.

It is based on Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren
T. (2017) On the Compactness, Efficiency, and Representation of 3D
Convolutional Networks: Brain Parcellation as a Pretext Task. In:
Niethammer M. et al. (eds) Information Processing in Medical Imaging.
IPMI 2017. Lecture Notes in Computer Science, vol 10265. Springer, Cham.
DOI: 10.1007/978-3-319-59050-9_28
"""

from keras.layers.convolutional import (Convolution3D, MaxPooling3D,
                                        ZeroPadding3D)
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Sequential, Model

def c3d():
    """
    Custom model for c3d.

    This function tries to modify the existing by 1) reducing it to 1 channel
    and, 2) actually changing the input to larger images.
    """
    # Load c3d base model
    c3d_model = c3d_base()
    final_model = c3d_model()


def c3d_base(weights=False, summary=True, weights_file=None):
    """
    Build base model for c3D using Seqential API.

    This model builds a base c3D network.
    """
    c3d_model = Sequential()
    # 1st layer group
    c3d_model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv1',
                                subsample=(1, 1, 1),
                                input_shape=(3, 16, 112, 112)))

    c3d_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))

    # 2nd layer group
    c3d_model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv2',
                                subsample=(1, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))

    # 3rd layer group
    c3d_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3a',
                                subsample=(1, 1, 1)))
    c3d_model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3b',
                                subsample=(1, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))

    # 4th layer group
    c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4a',
                                subsample=(1, 1, 1)))
    c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4b',
                                subsample=(1, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))

    # 5th layer group
    c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5a',
                                subsample=(1, 1, 1)))
    c3d_model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5b',
                                subsample=(1, 1, 1)))
    c3d_model.add(ZeroPadding3D(padding=(0, 1, 1)))
    c3d_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
    c3d_model.add(Flatten())

    # FC layers group
    c3d_model.add(Dense(4096, activation='relu', name='fc6'))
    c3d_model.add(Dropout(.5))
    c3d_model.add(Dense(4096, activation='relu', name='fc7'))
    c3d_model.add(Dropout(.5))
    c3d_model.add(Dense(487, activation='softmax', name='fc8'))

    if weights:
        c3d_model.load_weights(weights_file)

    # Remove last layer and add our own


    if summary:
        print(c3d_model.summary())


    return c3d_model
