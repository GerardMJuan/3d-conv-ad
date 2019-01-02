# 3d-conv-ad
Implementation and testing of 3D convolutional neural networks for AD diagnosis

This repository holds three different version of 3DCNN:

1. Pre-trained model: 3D Res Net

Note: Currently, due to small network, batch size should be 1. We would need to test to reduce this:
1.1- Reducing input Size
1.2- Trying to select larger gpu (how?)
1.3- Do multigpu training
1.4- Fit full images and don't do cropping
1.5 use cropping of only one axis ( smaller? isotropic/anisotropic?)

Based from Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren
T. (2017) On the Compactness, Efficiency, and Representation of 3D
Convolutional Networks: Brain Parcellation as a Pretext Task. In:
Niethammer M. et al. (eds) Information Processing in Medical Imaging.
IPMI 2017. Lecture Notes in Computer Science, vol 10265. Springer, Cham.
DOI: 10.1007/978-3-319-59050-9_28

2. Pre trained model: C3d


Note: Needs further testing and validation to be useful:
1. Different input Size
2. Different training parameters
3. Do cropping and do not fit full images

This model builds a base c3D network. http://vlg.cs.dartmouth.edu/c3d/


3. From scratch model (NYI)
