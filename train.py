"""
Main function that trains the model from input data.

This function gathers the data and feeds tensors into the network, training it.
"""

from model import c3d
from utils import crop_volume
from dataio import BrainSequence, slice_generator
from bids.grabbids import BIDSLayout
import pandas as pd
import configparser
import time
import argparse
import os

# Parser
def get_parser():
    parser = argparse.ArgumentParser(description='Training 3D conv network.')
    parser.add_argument("--config_file",
                        type=str, nargs=1, required=True, help='config file')
    parser.add_argument("--output_directory_name", type=str, nargs=1,
                        required=True, help='directory where the output will be')
    parser.add_argument("--crop", type=str, action="store_true",
                        help='wether to make new cropping or not.')

    return parser


def train(config_file, out_dir_name):
    """
    Main function for training.

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

    # The crops folder is a parent directory where all the
    # images are stored. The metadata file associates each
    # image to its label
    crop_metadata_file = config["data"]["metadata_crops"]
    df_metadata = None
    # Create crops (optional), if already done there is no need.
    if args.crop:
        # Load dataset and labels using BIDS
        # The metadata file contains information about
        # all the samples we will use. All the subjects mentioned in the
        # metadata must have a corresponding MRI image present
        # in the BIDS folder.
        bids_folder = config["data"]["bids_folder"]
        metadata_file = config["data"]["bids_folder"]
        layout = BIDSLayout(bids_folder)

        # For each entry in the metadata file
        rows_list = []
        for subj in metadata_file.itertuples():
            # locate the corresponding MRI scan
            ptid = subj["PTID"]
            ptid_bids = 'ADNI' + ptid[0:3] + 'S' + ptid[7:]
            # Hardcoded baselines
            file = layout.get(subject=ptid_bids, extensions='.nii.gz',
                              modality='anat', session='M00',
                              return_type='file')[0]
            # Actually perform the cropping
            size = config["general"]["size"]
            n_slices = config["general"]["n_slices"]
            new_crops = slice_generator(file, n_slices, size, out_dir)
            # Iterate over all the new crops
            for crop in new_crops:
                dict = {"path": crop,
                        "DX": subj["DX"]}
                rows_list.append(dict)
        # Save the new info about the image in df_crop
        df_crop = pd.DataFrame(rows_list)
        df_crop.to_csv(crop_metadata_file)

    else:
        df_metadata = pd.read_csv(crop_metadata_file)
        # Save metadata to disk

    # df_metadata does things

    # Create generator File

    # Initialize model

    # Train
    print('Procés finished.')
    t1 = time.time()
    print('Time to compute the script: ', t1 - t0)

    """
    # Load the model
    UNDER CONSTRUCTION, TODO
    model = c3d.model()
    mean = c3d.mean

    model = vgg16.model(weights=True, summary=True)
    mean = vgg16.mean
    model.compile(loss='mse', optimizer='sgd')
    X = X - mean
    model.fit(X, Y)
    """


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train(args.config_file[0], args.output_directory_name[0])
