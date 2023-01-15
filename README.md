# SuperResolution_using_UNET

This repository contains all scripts used to train a super resolution machine learning model based on the UNET architecture. 

A super resolution model is when you take an image with a low resolution (in my case, 64x64), upscale it (which results in a very blurry and undetailed image) and run that image through a pretrained model to add detail back in. For this project I chose to use a UNET architecture, which can be trained to perform tasks where the input and output data that can both be represented as images, such as super resolution, semantic segmentation and denoising.

If you want to create a super resolution model yourself, you will need to create a dataset first, which is just putting a ton of images in a folder. The training script will read these images at runtime and create appropriate input and output data to train on. I suggest using the Google open images dataset if you are looking for a model that generalises, or using the CreateDataset.py script if you want to create training images using your webcam.


