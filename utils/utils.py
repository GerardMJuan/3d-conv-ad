"""
Utils.

Other functions that are useful for the implementation.
"""

import SimpleITK as sitk


def resize_image(img, new_size):
    """
    Resample brain MRI image to specified spacing, size_out and spacing out.

    img: The MRI image to resample.
    new_size: The spacing of the image we want.

    Function adapted from CODE/scripts_py/resample_image.py
    """
    sz_in, sp_in = img.GetSize(), img.GetSpacing()
    or_in, dir_in = img.GetOrigin(), img.GetDirection()
    new_size = [int(x) for x in new_size]
    new_spacing = [old_sz*old_spc/new_sz for old_sz, old_spc, new_sz in
                   zip(sz_in, sp_in, new_size)]
    t = sitk.Transform(3, sitk.sitkScale)
    # TODO: IF NEEDED, ADD GAUSSIAN SMOOTHING
    out_sitk = sitk.Resample(img, new_size, t, sitk.sitkLinear,
                             or_in, new_spacing,
                             dir_in, 0.0, sitk.sitkFloat32)
    return out_sitk


def extract_representation(model, seq, layer):
    """
    Given a model and a generator of images, extract the representation of a given layer
    model: pretrained model
    seq: generator of images that are adecuate to the model
    layer: layer of the model we want to extract the representation

    return a list of representation equal to the number of images provided by the generator.
    """
    print('NYI!')
