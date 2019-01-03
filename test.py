"""
Main function that tests the results on a given trained model.
(NYI)
"""

import argparse
import os
import configparser
import time
import pandas as pd
import numpy as np
import bids.layout

def get_parser():
    """Parse data for main function."""
    parser = argparse.ArgumentParser(description='Testing 3D conv network.')
    parser.add_argument("--config_file",
                        type=str, nargs=1, required=True, help='config file')
    parser.add_argument("--output_directory_name", type=str, nargs=1,
                        required=True, help='directory where the output will be')

    return parser


def main(config_file, out_dir):
    """
    Test a trained model.

    Load the trained model, and test its performance.
    """


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.config_file[0], args.output_directory_name[0])
