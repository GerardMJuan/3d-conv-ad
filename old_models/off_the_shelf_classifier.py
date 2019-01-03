"""
Baseline for AD classification using CNN.

This file serves as a testing file for 1) make sure that the segmentation works (this means that the weight transfer was done correctly) and 2) to create a baseline for AD classification that serves as a parting point for future work. """

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
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
    Execute Main function for the classifier.

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

    # Preprocess labels
    label_dict = dict(zip(["NL", "MCI", "AD"], range(0, 3)))

    # GET CORRESPONDING DX AND PATHS OF SAID SUBJECTS
    X_train = df_metadata[df_metadata["PTID"].isin(S_train)].path.values
    Y_train = df_metadata[df_metadata["PTID"].isin(S_train)].DX.map(label_dict, na_action='ignore').values

    X_test = df_metadata[df_metadata["PTID"].isin(S_test)].path.values
    Y_test = df_metadata[df_metadata["PTID"].isin(S_test)].DX.map(label_dict, na_action='ignore').values

    # Create sequences of train/test (no really need for validation here)
    BrainSeq = BrainSequence(X_train, to_categorical(Y_train), batch_size, norm='hist',
                             norm_param=mean_file, train=True, crop=True)
    BrainSeq_test = BrainSequence(X_test, to_categorical(Y_test), batch_size, norm='hist',
                                  norm_param=mean_file, train=False, crop=True)

    # Load model

    model = HighRes3DNet_base(input_shape=(96, 96, 96, 1), weights=True, summary=True,
                              weights_dir=config['model']['weights'])

    # Extract representations and train the simple model
    img_train = extractRepresentation(model, BrainSeq)
    img_test = extractRepresentation(model, BrainSeq_test)

    ad_svm = SVC()

    ad_lr = LogisticRegression()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train(args.config_file[0], args.output_directory_name[0])
