"""
Implementation of neural network models.

Each function implements a different model
"""

from keras.layers.convolutional import (Conv3D, MaxPooling3D,
                                        ZeroPadding3D)
from keras import models
from keras import layers
from keras import utils
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
from contextlib import redirect_stdout

def c3d_base(input_shape=(16, 112, 112, 3), weights=False, weights_file=None):
    """
    Build base model for c3D using Seqential API.

    This model builds a base c3D network. http://vlg.cs.dartmouth.edu/c3d/
    """
    print(input_shape)
    c3d_model = models.Sequential()
    # 1st layer group
    c3d_model.add(Conv3D(64, (3, 3, 3), activation='relu',
                         name='conv1',
                         input_shape=input_shape,
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
    c3d_model.add(layers.Flatten())

    # FC layers group
    c3d_model.add(layers.Dense(4096, activation='relu', name='fc6'))
    c3d_model.add(layers.Dropout(.5))
    c3d_model.add(layers.Dense(4096, activation='relu', name='fc7'))
    c3d_model.add(layers.Dropout(.5))
    c3d_model.add(layers.Dense(487, activation='softmax', name='fc8'))

    if weights:
        c3d_model.load_weights(weights_file)

    # Create new model
    # Get input
    new_input = c3d_model.input
    # Find the layer to connect
    hidden_layer = c3d_model.layers[-2].output
    # Connect a new layer on it
    new_output = layers.Dense(3, activation='softmax', name='fc9')(hidden_layer)
    # Build a new model
    c3d_model_2 = models.Model(new_input, new_output)

    # Remove last layer and add our own

    # Fix all layers but last 3
    # for layer in c3d_model_2.layers[:17]:
    #     layer.trainable = False

    if summary:
        c3d_model_2.summary()
    return c3d_model_2


def padding(input_tensor, n_param):
    """
    Auxiliar function to pad 3d tensors.

    It is used to wrap it into a Lambda layer.
    """
    n_bypass = input_tensor.shape[-1]
    pad_1 = np.int((n_param - n_bypass) // 2)
    pad_2 = np.int(n_param - n_bypass - pad_1)
    padding_dims = np.vstack(([[0, 0]],
                              [[0, 0]] * 3,
                              [[pad_1, pad_2]]))
    input_tensor_pad = tf.pad(tensor=input_tensor,
                              paddings=padding_dims.tolist(),
                              mode='CONSTANT')
    return input_tensor_pad


# Building block of the HighRes3DNet
def res_layer(input_tensor, kernel_size, filter, d, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        d. amount of dilation
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        weights: initializing weights
    # Returns
        Output tensor for the block.
    """
    bn_axis = 4

    conv_name_base = 'res_' + str(stage) + "_" + str(block) + 'conv'
    bn_name_base = 'res_' + str(stage) + "_" + str(block) + '_bn'

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_0')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(filter, kernel_size,
                      kernel_initializer='he_normal',
                      padding='same',
                      strides=(1, 1, 1),
                      dilation_rate=d,
                      use_bias=False,
                      name=conv_name_base + '_0')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_1')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv3D(filter, kernel_size,
                      padding='same',
                      strides=(1, 1, 1),
                      dilation_rate=d,
                      use_bias=False,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '_1')(x)

    # Sum the layers, either by
    n_param = x.shape[-1]
    input_tensor_pad = layers.Lambda(padding, arguments = {"n_param": n_param})(input_tensor)
    x = layers.Add()([x, input_tensor_pad])

    return x


def HighRes3DNet_base(input_shape=(96, 96, 96, 1), weights=False, summary=True, weights_dir=None):
    """
    Tentative of HighRes3DNet model tranining, reimplemented in Keras.

    Based from Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren
    T. (2017) On the Compactness, Efficiency, and Representation of 3D
    Convolutional Networks: Brain Parcellation as a Pretext Task. In:
    Niethammer M. et al. (eds) Information Processing in Medical Imaging.
    IPMI 2017. Lecture Notes in Computer Science, vol 10265. Springer, Cham.
    DOI: 10.1007/978-3-319-59050-9_28
    """

    # Convert the weights
    # Only need to do it ONCE, already did so not important
    if weights:
        checkpoint_path = os.path.join(weights_dir, "model.ckpt-33000")
        # weights = convert_high3dresnet_weights(checkpoint_path)

    # The network consists of 20 layers of convolutions.
    img_input = layers.Input(shape=input_shape)
    x = layers.Conv3D(16, (3, 3, 3),
                      strides=(1, 1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      name='conv_0_bn_reluconv')(img_input)
    x = layers.BatchNormalization(axis=4, name='conv_0_bn_relu_bn')(x)
    x = layers.Activation('relu')(x)

    # Res blocks
    x = res_layer(x, [3, 3, 3], 16, d=1, stage=1, block=0)
    x = res_layer(x, [3, 3, 3], 16, d=1, stage=1, block=1)
    x = res_layer(x, [3, 3, 3], 16, d=1, stage=1, block=2)

    # Res blocks
    x = res_layer(x, [3, 3, 3], 32, d=2, stage=2, block=0)
    x = res_layer(x, [3, 3, 3], 32, d=2, stage=2, block=1)
    x = res_layer(x, [3, 3, 3], 32, d=2, stage=2, block=2)

    # Res blocks
    x = res_layer(x, [3, 3, 3], 64, d=4, stage=3, block=0)
    x = res_layer(x, [3, 3, 3], 64, d=4, stage=3, block=1)
    x = res_layer(x, [3, 3, 3], 64, d=4, stage=3, block=2)

    # Dropout
    x = layers.Conv3D(80, (1, 1, 1),
                      strides=(1, 1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      activation='softmax',
                      use_bias=False,
                      name='conv_1_bn_reluconv')(x)
    x = layers.BatchNormalization(axis=4, name='conv_1_bn_relu_bn')(x)
    x = layers.Activation('relu')(x)

    # Output file
    x = layers.Conv3D(160, (1, 1, 1),
                      strides=(1, 1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      use_bias=False,
                      name='conv_2_bnconv')(x)
    x = layers.BatchNormalization(axis=4, name='conv_2_bn')(x)
    x = layers.Activation('softmax')(x)
    model = models.Model(inputs=img_input, outputs=x)

    if weights:
        # model = load_weights(model, weights, weights_dir)
        # if we already have the weights saved to disK:
        model.load_weights(os.path.join(weights_dir, "weights_res3dnet.h5"))


def HighRes3DNet_cs(input_shape=(96, 96, 96, 1), weights=False, summary=True, weights_dir=None):
    """
    From the base model, create a variation for classification.
    """

    model = HighRes3DNet_base(input_shape, weights, summary, weights_dir)

    # Get input
    new_input = model.input
    # Find the layer to connect
    hidden_layer = model.layers[-4].output
    print(hidden_layer.name)
    x = layers.MaxPooling3D(pool_size=(3, 3, 3))(hidden_layer)
    x = layers.Flatten()(x)
    # x = layers.Dense(512, kernel_initializer='he_normal', name='fc0')(x)
    # x = layers.Conv1D(512, kernel_initializer='he_normal', name='fc0')(x)
    # x = layers.BatchNormalization(axis=-1, name='fc0_bn')(x)
    # x = layers.Dense(1024, kernel_initializer='he_normal', name='fc1')(x)
    # x = layers.BatchNormalization(axis=-1, name='fc1_bn')(x)
    x = layers.Dense(3, activation='softmax', name='fc2')(x)

    # Build a new model
    model_2 = models.Model(new_input, x)

    # Remove last layer and add our own

    # Fix all layers but last ones (variable)
    # for layer in model_2.layers[:-10]:
    #     layer.trainable = False

    if summary:
        with open('modelsummary_base.txt', 'w') as f:
            with redirect_stdout(f):
                utils.print_summary(model, line_length=110, print_fn=None)

        with open('modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                utils.print_summary(model_2, line_length=110, print_fn=None)
    return model_2



def convert_high3dresnet_weights(checkpoint_path):
    """
    Auxiliar function to convert tensorflow weights
    to keras weights.
    """
    weights = {}

    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    print('Load tensorflow variables')
    for key in var_to_shape_map:
        # All of these are weights that we do not want to use
        if key in ['beta2_power', 'beta1_power']:
            continue
        if 'Adam' in key:
            continue
        if 'ExponentialMovingAverage' in key:
            continue
        if '_local_step' in key:
            continue
        if 'biased' in key:
            continue
        if not reader.get_tensor(key).shape:
            continue

        # Create a more legible key
        key_n = key.replace('/', '_')

        # replace duplicate things
        key_n = key_n.replace('HighRes3DNet', '')
        key_n = key_n.replace('_bn_bn', '_bn')
        key_n = key_n.replace('__', '_')

        # Adapt variables names to new network
        key_n = key_n.replace('_res', 'res')
        key_n = key_n.replace('_conv', 'conv')

        key_n = key_n.replace('_w', '/kernel:0')
        key_n = key_n.replace('_biases', '/bias:0')
        key_n = key_n.replace('_moving_mean', '/moving_mean:0')
        key_n = key_n.replace('_moving_variance', '/moving_variance:0')
        key_n = key_n.replace('_moving_mean', '/moving_mean:0')
        key_n = key_n.replace('_gamma', '/gamma:0')
        key_n = key_n.replace('_beta', '/beta:0')

        print(key_n)
        weights[key_n] = reader.get_tensor(key)
    return weights


def load_weights(model, weights, model_dir):
    """
    Assign weights to layers.

    model: entire model of the network.
    weights: dictionary with the weights in numpy format.
    """
    print("Load weights")
    for layer in model.layers:
        if layer.weights:
            weights_n = []
            for w in layer.weights:
                print(w.name)
                # If bias, ignore it
                if '/bias:0' in w.name:
                    continue
                # Locate the corresponding array in weights dict
                try:
                    weight_arr = weights[w.name]
                except:
                    print('ERROR')
                    print(w.name)
                    continue
                weights_n.append(np.array(weight_arr))

            # layer.get_weights() components should have the same shape as weights_n
            for k1, k2 in zip(weights_n, layer.get_weights()):
                assert k1.shape == k2.shape
            layer.set_weights(np.array(weights_n))

    print('Saving model weights...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_weights(os.path.join(model_dir, "weights_res3dnet.h5"))
    return model

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes
