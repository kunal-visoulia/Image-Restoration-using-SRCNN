# Image-Restoration-using-SRCNN
Deploying the super-resolution convolution neural network (SRCNN) using Keras

This network was published in the paper, "Image Super-Resolution Using Deep Convolutional Networks" by Chao Dong, et al. in 2014. 
**Referenced Research Paper** https://arxiv.org/abs/1501.00092.

## SRCNN
The SRCNN is a deep convolutional neural network that learns end-to-end mapping of low resolution to high resolution images. As a result, we can use it to improve the image quality of low resolution images.

To evaluate the performance of this network, we will be using three image quality metrics:
- peak signal to noise ratio (PSNR), 
- mean squared error (MSE), and 
- structural similarity (SSIM) index.

We will also be using OpenCV to pre and post process our images. Also, there is frequent converting of our images back and forth between the RGB, BGR, and YCrCb color spaces. This is necessary because the SRCNN network was trained on the luminance (Y) channel in the YCrCb color space.

