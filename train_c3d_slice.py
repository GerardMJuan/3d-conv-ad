    # -*- coding: utf-8 -*-
"""
Main function that trains the model from input data, using a c3d pretrained net.

This function gathers the data and feeds tensors into the network, training it.
"""

from utils.model import c3d_base
from utils.dataio import make_crops, BrainSequence
import pandas as pd
import configparser
import time
import argparse
import os
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
import bids.layout


# Parser
def get_parser():
    """Parse data for main function."""
    parser = argparse.ArgumentParser(description='Training 3D conv network.')
    parser.add_argument("--config_file",
                        type=str, nargs=1, required=True, help='config file')
    parser.add_argument("--output_directory_name", type=str, nargs=1,
                        required=True, help='directory where the output will be')
    parser.add_argument("--crop", action="store_true",
                        help='wether to make new cropping or not.')

    return parser


def train(config_file, out_dir_name, crop):
    """
    Execute Main function for training.

    Trains the model with a given dataset.
    """
    t0 = time.time()
    # Load configuration
    # Load the configuration of a given experiment.
    config = configparser.ConfigParser()
    config.read(config_file)

    # Create output directory to store results
    out_dir = (config["folders"]["EXPERIMENTS"] +
               out_dir_name + os.sep)
    # Create out directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load tranining parameters
    batch_size = int(config["model"]["batch_size"])
    new_img_size = (int(x) for x in config["general"]["new_size"].split(','))
    epochs = int(config["model"]["epochs"])
    weights_file = config["model"]["weights"]
    mean_file = config["model"]["mean"]
    metadata_file = config["data"]["metadata"]
    bids_folder = config["data"]["bids_folder"]

    crops_dir = out_dir + '/' + 'CROPS' + '/'
    if not os.path.exists(crops_dir):
        os.makedirs(crops_dir)
    # The crops folder is a parent directory where all the
    # images are stored. The metadata file associates each
    # image to its label
    crop_metadata_file = out_dir + config["data"]["metadata_crops"]
    # Create crops (optional), if already done there is no need.
    if crop:
        # Load dataset and labels using BIDS
        # The metadata file contains information about
        # all the samples we will use. All the subjects mentioned in the
        # metadata must have a corresponding MRI image present
        # in the BIDS folder.
        bids_folder = config["data"]["bids_folder"]
        metadata_file = config["data"]["metadata"]
        new_size = config["general"]["new_size"].split(',')
        make_crops(bids_folder, metadata_file,
                   crops_dir, crop_metadata_file, new_size)

    # Divide between test, train and validation
    df_metadata = pd.read_csv(crop_metadata_file)

    # Get list of unique subjects
    subj = df_metadata.PTID.values
    dx = df_metadata.DX.values
    s = list(set(zip(subj, dx)))
    x, y = zip(*s)

    # Get train/test/val
    # THOSE ARE SUBJECT NAMES
    rd_seed = 1714
    S_train, S_test, DX_train, DX_test = train_test_split(x, y, test_size=.2, random_state=rd_seed)
    S_train, S_val, DX_train, DX_val = train_test_split(S_train, DX_train, test_size=.2, random_state=rd_seed)

    # Preprocess labels
    label_dict = dict(zip(["NL", "MCI", "AD"], range(0, 3)))

    # GET CORRESPONDING DX AND PATHS OF SAID SUBJECTS
    X_train = df_metadata[df_metadata["PTID"].isin(S_train)].path.values
    Y_train = df_metadata[df_metadata["PTID"].isin(S_train)].DX.map(label_dict, na_action='ignore').values

    X_valid = df_metadata[df_metadata["PTID"].isin(S_val)].path.values
    Y_valid = df_metadata[df_metadata["PTID"].isin(S_val)].DX.map(label_dict, na_action='ignore').values

    # Load neural network parameters
    batch_size = int(config["model"]["batch_size"])
    epochs = int(config["model"]["epochs"])
    weights_file = config["model"]["weights"]
    mean_file = config["model"]["mean"]
    mean = np.load(mean_file)
    # Crop mean
    mean = mean[:, :, 8:120, 30:142]
    print("mean shape")
    print(mean.shape)
    # Convert mean to black and white
    mean = np.mean(mean, axis=0)

    # Create generator File
    BrainSeq = BrainSequence(X_train, to_categorical(Y_train), batch_size, norm='mean',
                             norm_param=mean, color=True, train=True, crop=False, img_aug=False)
    BrainSeq_val = BrainSequence(X_valid, to_categorical(Y_valid), batch_size, norm='mean',
                             norm_param=mean, color=True, train=False, crop=False, img_aug=False)

    # Initialize model
    print(new_img_size)
    x, y, z = new_img_size
    model = c3d_base(input_shape=(x, y, z, 3), weights=True, weights_file=weights_file)
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, amsgrad=False)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    callb = TensorBoard(log_dir=out_dir + 'logs/', histogram_freq=0, batch_size=batch_size,
                        write_graph=True, write_grads=True, write_images=False,
                        embeddings_freq=0, embeddings_layer_names=None,
                        embeddings_metadata=None, embeddings_data=None)

    # use early stopping callback

    model.fit_generator(BrainSeq,
                        steps_per_epoch=None,
                        epochs=epochs,
                        shuffle=True,
                        callbacks=[callb],
                        verbose=1,
                        validation_data=BrainSeq_val)

    # Validate the model with a custom predictive function

    X_test = df_metadata[df_metadata["subj"].isin(S_test)].path.values
    Y_test = df_metadata[df_metadata["subj"].isin(S_test)].DX.map(label_dict, na_action='ignore').values

    BrainSeq_test = BrainSequence(X_test, to_categorical(Y_test), batch_size, norm='mean',
                             norm_param=mean, color=True, train=False, crop=False)

    # evaluate
    score = model.evaluate_generator(BrainSeq_test)
    print(score)

    print('Proces finished.')
    t1 = time.time()
    print('Time to compute the script: ', t1 - t0)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train(args.config_file[0], args.output_directory_name[0], args.crop)
