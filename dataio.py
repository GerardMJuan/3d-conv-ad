"""
Data in-out.

FUnctions that deal with input and output of data and conversion to tensors.

Most of the data in/out funcionality is gathered from library DLTK:
https://github.com/DLTK
"""
import SimpleITK as sitk
import os
import numpy as np
from utils import resize_image
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

    def __init__(self, x_set, y_set, batch_size, mean):
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
        self.mean = mean

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

            # Data preparation. sort from less to more, should work! (maybe)
            # Substract the mean of the trained set from the base images
            img = img - self.mean

            # duplicate the img onto the three channelsb = np.repeat(a[:, :, np.newaxis], 3, axis=2)
            img = np.repeat(img[:, :, :, np.newaxis], 3, 3)
            print(img.shape)
            # Data augmentation:
            # TODO:

            batch_img.append(img)
        return np.array(batch_img), np.array(batch_y)


def make_crops(bids_folder, metadata_file, out_dir, out_file, new_size):
    """
    Create a new dataset of crops from an existing dataset.

    Given a folder of images, and a csv file containing the info
    about the dataset, this function makes random crops of the images and
    save them to disk
    bids_folder: folder where the images are stored (in BIDS format)
    metadata_file: path to csv file where the info about images is stored.
    out_dir: directory where to save the cropped images
    out_file: file where to store the info about the crops
    resample_size: size to resample the images.
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
        new_crops = slice_generator(file, out_dir, new_size)
        # Iterate over all the new crops
        for crop in new_crops:
            dict = {"path": crop,
                    "subj": subj.PTID,
                    "DX": subj.DX}
            rows_list.append(dict)
    # Save the new info about the image in df_crop
    df_crop = pd.DataFrame(rows_list)
    df_crop.to_csv(out_file)


def slice_generator(image_file, out_dir, new_size=None):
    """
    Image generator.

    creates slices of a given size for an image and saves them
    to disk.
    image_filename: Filename of the image.
    out_dir: output directory.

    Returns the paths of each of the saved slices.
    """
    # for each image file
    crop_paths = []
    # load img
    img = load_img(image_file, new_size)
    # current pointer of slice
    i = 0
    while i + 16 < 112:
        # For each dimension
        # also transpose it so that the small dim goes first
        i1 = img[:, :, i:i+16]
        img_cropped_1 = np.transpose(i1, (2, 0, 1))
        i2 = img[:, i:i+16, :]
        img_cropped_2 = np.transpose(i2, (1, 0, 2))
        img_cropped_3 = img[i:i+16, :, :]
        print(img_cropped_1.shape)
        print(img_cropped_2.shape)
        print(img_cropped_3.shape)
        i = i + 6
        out_file = out_dir + os.path.basename(image_file) + '_crop' + str(i)
        # Save paths and images (only if it has anything else than 0)
        if np.any(img_cropped_1):
            save_img(img_cropped_1, out_file + '_1.nii.gz')
            crop_paths.append(out_file + '_1.nii.gz')
        if np.any(img_cropped_2):
            save_img(img_cropped_2, out_file + '_2.nii.gz')
            crop_paths.append(out_file + '_2.nii.gz')
        if np.any(img_cropped_3):
            save_img(img_cropped_3, out_file + '_3.nii.gz')
            crop_paths.append(out_file + '_3.nii.gz')

    return crop_paths


def save_img(img, path):
    """
    Save a given image to the path specified.

    img: image to save.
    path: path where to save the image.
    """
    img_out = sitk.GetImageFromArray(img)
    sitk.WriteImage(img_out, path)


def load_img(path, new_size=None):
    """
    Load a single image from disk.

    path: Image path
    resample_spacing: spacing to resample img
    """
    sitk_t1 = sitk.ReadImage(path)
    # if we have a resample size:
    if new_size:
        sitk_t1 = resize_image(sitk_t1, new_size)
    img = sitk.GetArrayFromImage(sitk_t1)
    save_img(img, '/homedtic/gmarti/test.nii.gz')
    return img
