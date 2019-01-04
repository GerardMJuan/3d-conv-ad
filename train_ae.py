"""
Train a basic CNN.
"""
import argparse
import os
import configparser
import time
import pandas as pd
import numpy as np
import bids.layout
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import TerminateOnNaN
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.cross_validation import train_test_split
from utils.models import AE3D
from utils.dataio import BrainSequence

# Parser
def get_parser():
    """Parse data for main function."""
    parser = argparse.ArgumentParser(description='Training 3D conv network.')
    parser.add_argument("--config_file",
                        type=str, nargs=1, required=True, help='config file')
    parser.add_argument("--output_directory_name", type=str, nargs=1,
                        required=True, help='directory where the output will be')

    return parser


def main(config_file, out_dir_name):
    """
    Execute Main function for training.

    Trains the model with a given dataset.
    """
    t0 = time.time()
    rd_seed = 1714
    np.random.seed(rd_seed)
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
    metadata_file = config["data"]["metadata"]
    bids_folder = config["data"]["bids_folder"]

    # Load BIDS layout
    layout = bids.layout.BIDSLayout([(bids_folder, 'bids')])

    ## Load data (THIS NEED TO BE CHANGED)
    # ALL THE DATA LOADING MUST BE CHANGED
    # Divide between test, train and validation
    df_metadata = pd.read_csv(metadata_file)

    # add a new column with the path of each file
    paths = []
    for subj in df_metadata.itertuples():
        # locate the corresponding MRI scan
        ptid = subj.PTID

        # If it is not NL or AD, ignore
        if subj.DX not in ['NL', 'AD']:
            paths.append(np.nan)
            continue
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
    x, y = zip(*s)

    # Get train/test/val
    # THOSE ARE SUBJECT NAMES
    S_train, S_test, DX_train, DX_test = train_test_split(x, y, test_size=.1, random_state=rd_seed)

    # Preprocess labels
    label_dict = dict(zip(["NL", "AD"], range(0, 2)))

    # GET CORRESPONDING DX AND PATHS OF SAID SUBJECTS
    X_train = df_metadata[df_metadata["PTID"].isin(S_train)].path.values
    Y_train = df_metadata[df_metadata["PTID"].isin(S_train)].DX.map(label_dict, na_action='ignore').values

    X_test = df_metadata[df_metadata["PTID"].isin(S_test)].path.values
    Y_test = df_metadata[df_metadata["PTID"].isin(S_test)].DX.map(label_dict, na_action='ignore').values

    # Create sequences of train/test (no really need for validation here)
    BrainSeq = BrainSequence(X_train, to_categorical(Y_train), batch_size, norm='none', train=True, crop=False, new_size=(193, 229, 193))
    BrainSeq_val = BrainSequence(X_test, to_categorical(Y_test), batch_size, norm='none', train=False, crop=False, new_size=(193, 229, 193))

    # Load data (THIS NEEDS TO BE CHANGED)

    # Create model
    # First output is layer, next is image
    model = AE3D(input_shape=(193, 229, 193, 1))

    opt = Adam(lr=0.0001)
    # Compile model

    model.compile(optimizer=opt,
                  loss=['categorical_crossentropy', 'mean_squared_error'],
                  metrics=['accuracy'])

    # Create callbacks
    # Early stopping for reducing over-fitting risk
    stopper = EarlyStopping(patience=1)

    # Model checkpoint to save the training results
    checkpointer = ModelCheckpoint(
        filepath=out_dir + "model_trained.h5",
        verbose=0,
        save_best_only=True,
        save_weights_only=True)

    # CSVLogger to save the training results in a csv file
    csv_logger = CSVLogger(out_dir + 'csv_log.csv', separator=';')

    # Callback to reduce learning rate
    def lr_scheduler(epoch, lr):
        if epoch == 15:
            return lr
        elif epoch == 25:
            return lr*.1
        elif epoch == 35:
            return lr*.1
        else:
            return lr

    lrs = LearningRateScheduler(lr_scheduler)

    # Callback to terminate on NaN loss (so terminate on error)
    NanLoss = TerminateOnNaN()

    callbacks = [checkpointer, csv_logger, stopper, NanLoss, lrs]

    # Train model
    model.fit_generator(BrainSeq,
                        steps_per_epoch=None,
                        epochs=epochs,
                        shuffle=True,
                        callbacks=callbacks,
                        verbose=1,
                        validation_data=BrainSeq_val)

    # Model is saved due to callbacks

    print('The end.')
    t1 = time.time()
    print('Time to compute the script: ', t1 - t0)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.config_file[0], args.output_directory_name[0])
