"""
Script to preprocess MRI images to make them ready for use in a CNN.

Need to implement:
1) Make sure that all the volumes have the same size.
1) Remove 10 slices from start/end of the volume in coronal view.
2) Apply a volume-wise Gaussian Normalization.
3) Do some kind of data augmentation? (LATER)
4) Divide in train/test/and validation using 8:1:1 split.
5) Save them to disk, and store their metadata related information.

Also, currently using registered, no need for that!
"""

# Data can be found in directory: /homedtic/gmarti/DATA/Data/ADNI_BIDS_STANDARD
# But do not use derivatives (ADD SAMPLE CODE)

# Load the data

# Resize
