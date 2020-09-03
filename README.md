# MsDNN
This is an official implementation of "**Multi-scale deep neural networks for real image super-resolution**" via TensorFlow

## Dependencies
- Tensorflow-gpu==1.9
- Python==2.7

## Illustration
- `MsDNN/model/msdnn_feed_v1_96a96_64blocks_1000000`: a multi-scale model with 64 residual blocks
- **Warning**: The file `model.checkpoint-999999.data-00000-of-00001` in the fold `msdnn_feed_v1_96a96_64blocks_1000000`
is large, one should download it alone and put into the corresponding folder.
- `MsDNN/RealSR/Test_LR`: The testing dataset provided by the NTIRE2019 SR challenge
- `MsDNN/msdnn.py`: The detailed structure of MsDNN
- `MsDNN/msdnn_demo.py`: The codes of obtaining high-resolution images

## Quick test
Commands of getting high-resolution images:
```python
python2 msdnn_demo.py
```
After executing the above command, there will exist a folder `MsDNN/RealSR/Test_HR`, which is the
super-resolution of testing dataset.

## Citation
If you find our work useful in your research or publication, please cite our work:

[1] Shangqi Gao, and Xiahai Zhuang, "**Multi-scale deep neural networks for real image super-resolution**", CVPR Workshops, 2019. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9025686)] [[arXiv](https://arxiv.org/pdf/1904.10698.pdf)]

```
@INPROCEEDINGS{msdnn/cvprw/2019, 
  author={S. {Gao} and X. {Zhuang}}, 
  booktitle={2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},   
  title={Multi-Scale Deep Neural Networks for Real Image Super-Resolution},   
  year={2019}, 
  pages={2006-2013}
}
```
