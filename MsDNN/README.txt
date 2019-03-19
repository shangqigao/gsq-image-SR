Method: A multi-scale deep neural networks (MsDNN) for real image super-resolution

Author: Shangqi Gao, Dengqiang Jia, Fuping Wu, Kaiwen Wan, Xiahai Zhuang*

Affiliation: School of Data Science, Fudan University

URL: https://github.com/shangqigao/NTIRE2019-image-SR

--------------------------------------------------------------------------------------

Instruction of files:

MsDNN/model/msdnn_feed_v1_96a96_64blocks_1000000: a multi-scale model with 64 residual blocks

MsDNN/RealSR/Test_LR: The testing dataset provided by the NTIRE2019 SR challenge

MsDNN/msdnn.py: The detailed structure of MsDNN

MsDNN/msdnn_demo.py: The codes of obtaining high-resolution images

---------------------------------------------------------------------------------------

Instruction of languages:

Tensorflow: GPU version

Python: 2.7

---------------------------------------------------------------------------------------

Commands of getting high-resolution images:

python2 msdnn_demo.py

After executing the above command, there will exist a fold MsDNN/RealSR/Test_HR, it is the
resolution of testing dataset



