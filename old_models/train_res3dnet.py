# -*- coding: utf-8 -*-
"""
Main function that trains the model from input data, using a pretrained
3DresNet.

This function gathers the data and feeds tensors into the network, training it.
module load cuDNN/7.0.5-CUDA-9.0.176
module load CUDA/9.0.176
"""

from utils.model import HighRes3DNet_base
from utils.dataio import BrainSequence, load_img
import pandas as pd
import configparser
import time
import argparse
import os
import numpy as np
import bids.layout
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard

# Parser
def get_parser():
    """Parse data for main function."""
    parser = argparse.ArgumentParser(description='Training 3D conv network.')
    parser.add_argument("--config_file",
                        type=str, nargs=1, required=True, help='config file')
    parser.add_argument("--output_directory_name", type=str, nargs=1,
                        required=True, help='directory where the output will be')

    return parser


def train(config_file, out_dir_name):
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
    epochs = int(config["model"]["epochs"])
    weights_file = config["model"]["weights"]
    mean_file = config["model"]["mean"]
    metadata_file = config["data"]["metadata"]
    bids_folder = config["data"]["bids_folder"]

    # Load BIDS layout
    layout = bids.layout.BIDSLayout([(bids_folder, 'bids')])

    # Divide between test, train and validation
    df_metadata = pd.read_csv(metadata_file)

    # add a new column with the path of each file
    paths = []
    for subj in df_metadata.itertuples():
        # locate the corresponding MRI scan
        ptid = subj.PTID
        ptid_bids = 'ADNI' + ptid[0:3] + 'S' + ptid[6:]
        # Hardcoded baselines
        try:
            file = layout.get(subject=ptid_bids, extensions='.nii.gz',
                              modality='anat', session='M00',
                              return_type='file')[0]
            paths.append(file)
        except:
            print('Ignoring subject ' + ptid)
            paths.append(np.nan)

    # remove subjects with missing entries
    df_metadata['path'] = paths
    df_metadata = df_metadata.dropna()
    print(len(df_metadata))

    # Get list of unique subjects
    subj = df_metadata.PTID.values
    dx = df_metadata.DX.values
    s = list(set(zip(subj, dx)))
    # DEBUGGING: SELECT SMALL AMOUNT OF THINGS
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

    X_test = df_metadata[df_metadata["PTID"].isin(S_test)].path.values
    Y_test = df_metadata[df_metadata["PTID"].isin(S_test)].DX.map(label_dict, na_action='ignore').values

    # Test: create list of images
    # X_train_img = [load_img(x) for x in X_train]
    # X_train_val = [load_img(x) for x in X_valid]
    # Create generator File
    BrainSeq = BrainSequence(X_train, to_categorical(Y_train), batch_size, norm='hist',
                             norm_param=mean_file, train=True, crop=True)
    BrainSeq_val = BrainSequence(X_valid, to_categorical(Y_valid), batch_size, norm='hist',
                                 norm_param=mean_file, train=False, crop=True)

    # Initialize model
    model = HighRes3DNet_cs(input_shape=(96, 96, 96, 1), weights=True, summary=True,
                              weights_dir=config['model']['weights'])
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, amsgrad=False)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # Train
    callb = TensorBoard(log_dir=out_dir + 'logs/', histogram_freq=0, batch_size=batch_size,
                        write_graph=True, write_grads=False, write_images=False,
                        embeddings_freq=0, embeddings_layer_names=None,
                        embeddings_metadata=None, embeddings_data=None)

    model.fit_generator(BrainSeq,
                        steps_per_epoch=None,
                        epochs=epochs,
                        shuffle=True,
                        callbacks=[callb],
                        verbose=1,
                        validation_data=BrainSeq_val)

    # TODO: Validate the model with a custom predictive function
    BrainSeq_test = BrainSequence(X_test, to_categorical(Y_test), batch_size, norm='hist',
                                  norm_param=mean_file, train=False, crop=True)

    # evaluate
    score = model.evaluate_generator(BrainSeq_test)
    print(score)

    print('Proces finished.')
    t1 = time.time()
    print('Time to compute the script: ', t1 - t0)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train(args.config_file[0], args.output_directory_name[0])
