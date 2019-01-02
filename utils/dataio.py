"""
Data in-out.

FUnctions that deal with input and output of data and conversion to tensors.

Most of the data in/out funcionality is gathered from library DLTK:
https://github.com/DLTK
"""
import SimpleITK as sitk
import os
import numpy as np
from utils.utils import resize_image
from keras.utils import Sequence
import pandas as pd
import bids.layout
from utils.hist_std import transform_by_mapping, read_mapping_file
import scipy.ndimage


class BrainSequence(Sequence):
    """
    Implement the Generator to train the model.


    Based on keras.utils.Sequence()
    """

    def __init__(self, x_set, y_set, batch_size, norm=None, norm_param=None,
                 color=False, train=True, img_aug=True, crop=False, new_size=None):
        """
        Initialize class.

        x_set: list of paths to the images.
        y_set: associated classes.
        batch_size: Size of the training batch.
        norm: to normalize each image or not. Could be None, 'mean' or 'hist'
        norm_param: parameters for normalization
        color: if we adapt the image to a three channel tensor
        crop: crop at generator time
        x_set and y_set should have same length
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.norm = norm
        self.norm_param = norm_param
        self.color = color
        self.training = train
        self.img_aug = img_aug
        self.crop = crop
        self.new_size = new_size

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
            # img = file
            img = load_img(file, self.new_size)
            if self.norm == 'mean':
                # Substract the mean of the trained set from the base images
                img = img - self.norm_param
            elif self.norm == 'hist':
                # Convert image to 5d
                img = img[np.newaxis, :, :, :, np.newaxis]

                # Histogram normalization
                # Create mask (will be full image)
                image_mask = np.ones_like(img, dtype=np.bool)
                # load mapping
                mapping = read_mapping_file(self.norm_param)
                # set cutoff
                cutoff = (0.001, 0.999)
                # transform img
                img = transform_by_mapping(img, image_mask, mapping['Modality0'], cutoff, type_hist = 'percentile')

                # return to 3d
                img = img[0, :, :, :, 0]

            # White normalization:
            img = whitening_transformation(img)

            # Crop image!
            # We return to three dimensions here!
            if self.crop:
                # Hardcoded, bad idea! Should not be! Confusing!
                img = generate_random_cropping(img, (96, 96, 96))

            # Image augmentation
            if self.training and self.img_aug:
                # rotation in any of the three angles
                img = rd_rotation(img, (-10, 10))
                # scale
                # DOESNT WORK, NEED FIX
                # img = rd_scale(img, (-10, +10))

            img = img[:, :, :, np.newaxis]
            if self.color:
                img = np.repeat(img, 3, 3)
            batch_img.append(img)

        # concatenate arrays by 0 axis
        batch_img = np.stack(batch_img, axis=0)
        return batch_img, np.array(batch_y)


def whitening_transformation(img):
    # make sure image is a monomodal volume
    img = (img - img.mean()) / max(img.std(), 1e-5)
    return img


def rd_rotation(img, range):
    """
    Creates a random rotation of a 3d matrix.

    Uses the specified range and in the three dimensions.
    """
    assert img.ndim == 3
    min_angle = float(range[0])
    max_angle = float(range[1])
    # Generate random angles between the range
    angle_x = np.random.uniform(min_angle,
                                max_angle) * np.pi / 180.0
    angle_y = np.random.uniform(min_angle,
                                max_angle) * np.pi / 180.0
    angle_z = np.random.uniform(min_angle,
                                max_angle) * np.pi / 180.0

    # Create the transformation matrix
    transform_x = np.array([[np.cos(angle_x), -np.sin(angle_x), 0.0],
                            [np.sin(angle_x), np.cos(angle_x), 0.0],
                            [0.0, 0.0, 1.0]])
    transform_y = np.array([[np.cos(angle_y), 0.0, np.sin(angle_y)],
                            [0.0, 1.0, 0.0],
                            [-np.sin(angle_y), 0.0, np.cos(angle_y)]])
    transform_z = np.array([[1.0, 0.0, 0.0],
                            [0.0, np.cos(angle_z), -np.sin(angle_z)],
                            [0.0, np.sin(angle_z), np.cos(angle_z)]])
    transform = np.dot(transform_z, np.dot(transform_x, transform_y))

    # Apply the affine transformation
    center_ = 0.5 * np.asarray(img.shape, dtype=np.int64)
    c_offset = center_ - center_.dot(transform)
    img = scipy.ndimage.affine_transform(
        img, transform.T, c_offset, order=3)
    return img


def rd_scale(img, ranges):
    """
    Randomly scale an image for a given range.
    """
    rand_zoom = np.random.uniform(low=ranges[0],
                                  high=ranges[1],
                                  size=(3,))
    rand_zoom = (rand_zoom + 100.0) / 100.0

    img = scipy.ndimage.zoom(img, np.array(rand_zoom), order=3)
    return img


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
    layout = bids.layout.BIDSLayout([(bids_folder, 'bids')])

    # For each entry in the metadata file
    rows_list = []
    for subj in df_metadata.itertuples():
        # locate the corresponding MRI scan
        ptid = subj.PTID
        ptid_bids = 'ADNI' + ptid[0:3] + 'S' + ptid[6:]
        # Hardcoded baselines
        try:
            file = layout.get(subject=ptid_bids, extensions='.nii.gz',
                              modality='anat', session='M00',
                              return_type='file')[0]
        except:
            print('Ignoring subject ' + ptid)
        # Actually perform the cropping
        new_crops = slice_generator(file, out_dir, new_size)
        # Iterate over all the new crops
        for crop in new_crops:
            dict = {"path": crop,
                    "PTID": subj.PTID,
                    "DX": subj.DX}
            rows_list.append(dict)
    # Save the new info about the image in df_crop
    df_crop = pd.DataFrame(rows_list)
    df_crop.to_csv(out_file)


def generate_random_cropping(img, size):
    """
    Create a random cropping from an input image, for a fixed size.

    The crop will be cubic.
    img: the image, in numpy. 3 dimensional.
    size: the size of the crop.
    """
    # get new coordinates
    # x y z could not coincide with actual x y z, just saying
    x = np.random.randint(0, img.shape[0]-size[0])
    y = np.random.randint(0, img.shape[1]-size[0])
    z = np.random.randint(0, img.shape[2]-size[0])
    i1 = img[x:x+size[0], y:y+size[0], z:z+size[0]]
    return i1


def slice_generator(image_file, out_dir, new_size=None):
    """
    Image generator.

    creates slices of a given size for an image and saves them
    to disk, to use for c3d.
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
