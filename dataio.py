"""
Data in-out.

FUnctions that deal with input and output of data and conversion to tensors.

Most of the data in/out funcionality is gathered from library DLTK:
https://github.com/DLTK
"""
import SimpleITK as sitk
import os
import numpy as np
from utils import crop_volume
from keras.utils import Sequence
import pandas as pd
from bids.grabbids import BIDSLayout

"""
import tensorflow as tf
import pandas as pd
import time
from matplotlib import pyplot as plt
from dltk.io.augmentation import *
from dltk.io.preprocessing import *
"""


class BrainSequence(Sequence):
    """
    Implement the Generator to train the model.

    Based on keras.utils.Sequence()
    """

    def __init__(self, x_set, y_set, batch_size, **kwargs):
        """
        Initialize class.

        x_set: list of paths to the images.
        y_set: associated classes.
        batch_size: Size of the training batch.
        n_slices: number of slices to get from each image.
        args: parameters for data augmentation
        x_set and y_set should have same length
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.kwargs = kwargs

    def __len__(self):
        """Return length of the steps per epoch."""
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Return a full batch of data for training.

        This procedure must also do data augmentation steps.
        idx: internal parameter
        """
        # Get indices of each image
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_img = []
        for file in batch_x:
            img = load_img(file)

            # Data preparation

            # Substract the mean of the trained set from the base images bla bla
            # TODO
            # Duplicate the image in the three channels.
            # # TODO:
            # Data augmentation:
            if self.args["Gaussian noise"]:
                # Apply gaussian noise with a given probability
                print(1)
            if self.args["Flipped"]:
                # Flip images with a given probability
                print(2)

            batch_img.append(img)
        return np.array(batch_img), np.array(batch_y)


def make_crops(bids_folder, metadata_file, n_slices, size, out_dir, out_file):
    """
    Create a new dataset of crops from an existing dataset.

    Given a folder of images, and a csv file containing the info
    about the dataset, this function makes random crops of the images and
    save them to disk
    bids_folder: folder where the images are stored (in BIDS format)
    metadata_file: path to csv file where the info about images is stored.
    n_slices: number of crops to do
    size: size of the crops, in 3D
    out_dir: directory where to save the cropped images
    out_file: file where to store the info about the crops
    """
    df_metadata = pd.read_csv(metadata_file)
    layout = BIDSLayout(bids_folder)

    # For each entry in the metadata file
    rows_list = []
    for subj in df_metadata.itertuples():
        # locate the corresponding MRI scan
        ptid = subj.PTID
        ptid_bids = 'ADNI' + ptid[0:3] + 'S' + ptid[6:]
        # Hardcoded baselines
        file = layout.get(subject=ptid_bids, extensions='.nii.gz',
                          modality='anat', session='M00',
                          return_type='file')[0]
        # Actually perform the cropping
        new_crops = slice_generator(file, n_slices, size, out_dir)
        # Iterate over all the new crops
        for crop in new_crops:
            dict = {"path": crop,
                    "DX": subj.DX}
            rows_list.append(dict)
    # Save the new info about the image in df_crop
    df_crop = pd.DataFrame(rows_list)
    df_crop.to_csv(out_file)


def slice_generator(image_file, n_slices, size, out_dir):
    """
    Image generator.

    creates slices of a given size for an image and saves them
    to disk.
    image_filename: Filename of the image.
    n_slices: number of slices.
    size: size of the slices.
    out_dir: output directory.

    Returns the paths of each of the saved slices.
    """
    # for each image file
    crop_paths = []
    # load img
    img = load_img(image_file)
    # For each slice
    for n in range(n_slices):
        # Crop the image
        img_crop = crop_volume(img, size, mode="Random")
        # Create output dir
        out_file = out_dir + os.path.basename(image_file) + '_crop' + str(n) + '.nii.gz'
        crop_paths.append(out_file)
        save_img(img_crop, out_file)
    return crop_paths


def save_img(img, path):
    """
    Save a given image to the path specified.

    img: image to save.
    path: path where to save the image.
    """
    img_out = sitk.GetImageFromArray(img)
    sitk.WriteImage(img_out, path)


def load_img(path):
    """
    Load a single image from disk.

    path: Image path
    """
    sitk_t1 = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(sitk_t1)
    return img
