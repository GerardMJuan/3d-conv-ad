"""
Main function that trains the model from input data.

This function gathers the data and feeds tensors into the network, training it.
"""

from model import c3d
from utils import crop_volume
from dataio import BrainSequence, slice_generator
import configparser
import time
import argparse

# Parser
def get_parser():
    parser = argparse.ArgumentParser(description='Training 3D conv network.')
    parser.add_argument("--config_file",
                        type=str, nargs=1, required=True, help='config file')
    parser.add_argument("--output_directory_name", type=str, nargs=1,
                        required=True, help='directory where the output will be')

    return parser

def train(config_file, out_dir_name):
    """
    Main function for training.

    Trains the model with a given dataset.
    """
    # Load configuration

    # Load dataset and labels

    # Create crops (optional), if already done there is no need.

    # Create generator File

    # Initialize model

    # Train


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
