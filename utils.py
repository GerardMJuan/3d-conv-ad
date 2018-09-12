"""
Utils.

Other functions that are useful for the implementation.
"""

import random


def crop_volume(img, size, mode="Random"):
    """
    Crop an slice of a fixed size from an image.

    img: image to crop.
    size: size of the cropping:
    mode: Type of cropping. Modes supported:
        Random: random cropping, that fits inside the image.
    """
    if mode == "Random":
        # size of img
        x, y, z = img.shape
        x_crop, y_crop, z_crop = [int(x) for x in size.split(',')]
        # the initial point can go from 0 to x - x_crop.
        new_x = random.randint(0, x-x_crop-1)
        new_y = random.randint(0, y-y_crop-1)
        new_z = random.randint(0, z-z_crop-1)

        img_cropped = img[new_x:new_x+x_crop,
                          new_y:new_y+y_crop,
                          new_z:new_z+z_crop]

        return img_cropped

    else:
        raise ValueError("Mode of cropping not supported!")
