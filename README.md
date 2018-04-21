# GAN
Simple implementation of GAN (Generative Adversarial Network), which is one of the deep learning model to generate image.  
This is implemented based on framework Tensorflow.

## Overview
This project consists of three programs.

* _GAN.py_  
    Main program of this project, and this is the imeplementation of GAN.

* _Trainer.py_  
    Sample program to run training. It requires training data images, and will be described below for details.

* _Predictor.py_
    Sample program to predict by using trained data. It requires trained parameter data and input image for prediction. This will also be described below.

## GAN.py
Python class which has training and prediction functions of GAN.
This class is implemented based on framework Tensorflow.

This file includes three classes: Generator, Disciminator and GAN.
GAN class is simple and standard implementation to combine Generator and Discriminator.
You can use GAN class only or implement your own class to combine Generator and Discriminator. 

_TODO:to be written about more technical details._


### Learning data
This implementation requires CSV file listing 2 image file paths in each row as learning data. File of the first column path is input for GAN, and the second column is file path of training data image. CSV is necessary because batch data read function of Tensorflow needs it. This repository include sample csv, dataList.csv, and image files as samples.

### Limitations
Currently, this implementation has some limitations.

* Treat only 1-dimensional image (e.g. gray scale image).
  There is a plan to extend it to treat 3-dimensional image (e.g. RGB color image), but not yet implemented.
* Data images have to be 512x512 pixels size._

### Customize and Settings
There are several settings.

* Optional argument of GAN constructor:
    - 'batch_size' is a number of image file pairs passed to GAN for each iteration of training. When predicting, this value should be 1.
    - 'is_data_augmentation' is a flag to use data augmentation while training. When this flag is 'True', input training image is augmented by flipping and transposing.
    - 'is_skip_connection' is a flag to use 'skip connection' network structure. This structure connect encoder to decoder in the same level directory in Generator. Thus detailed information of input image is kept while training.
    - 'loss_function' is a function to be used as loss function in Generator. The loss calculated by this function is mixed with output of Discriminator.

* Implementation of GAN class. You can use your own implementation to combine Generator and Discriminator class instead of GAN class.

_TODO:to be written more_


### Output data while learning
While learning, the implementation output the status of learning as below.

#### Standart output
Running part will write learning progress like below.  
_Step: 600, Loss: 90405240.000000 @ 2017/05/12 08:50:08_

This consists of 3parts:  
* Step value of learning process
* Loss value, which is sequare sum of pixel value differences between predicted values by generator and expected values
* Timestamp when the progress is output  

Default implementation of running parts write progress at the first step and every 200 steps after the first step.

#### Images predicted
_TODO:to be written_

#### Tensor Board
_GAN.py_ outputs files for _Tensor Board_ in _board_ directory.  
You can see _Tensor Board_ by _Tensorflow_'s usual way to invoke it.  

_TODO:to be written for more details._

#### Session file to resume learning
_Trainer.py_ saves files of session files, which includes trained parameters at the saving step.  
These files are saved in _saved_session_ directory.

_Trainer.py_ can load them to resume training from saved step.  

_TODO: Implement prediction program _Predictor.py_ , and write about it.  

## Trainer.py
Sample implementation of training by using GAN.py.
This implementation use data from  _dataList.csv_.  


_TODO:to be written about more technical details._

## Predictor.py
Sample implementation of prediction by using GAN.py.
This implementation use data from  _predictList.csv_.  

_TODO: Write technical details on it.  


## How to execute
1. Please run _GAN.py_, and class definitions are loaded.<br>
2. Please run _Trainer.py_. This program load training data and output session data of Tensorflow for every 200 training steps.
3. Please modify settings in _Predictor.py_, and run. This program load session data and output predicted image by matplotlib.

To run, libraries as stated below in _Requirements_ is necessary.

## Requirements
* Python3, Tensorflow 1.1.0, and the libraries it requires.
* _Trainer.py_ and _Predictor.py_ require _matplotlib_ to draw predicted images.
