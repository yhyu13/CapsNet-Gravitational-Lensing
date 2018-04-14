# Predict Gravitational Lensing with Capusle Network

---

## Introduction

Author: Joshua Yao-Yu Lin\*, Hang Yu\*, Gilbert Holder\*

    A capsule network [1] is a new type of neural network proposed by Geoffrey Hinton. Hinton argues that the convolution neural network has several problems; for example, they cannot handle the orientation information very well. With a capsule network, the neurons are gathered in a group called capsule, and they output a “vector” instead of scalars.

![Figure 1. Architecture of Capsule Network with dynamic routing](/imgs/Screen Shot 2018-04-06 at 1.56.46 PM.png)

Therefore, a capsule network can remember the “pose” of the object in images. For example, even when digits are overlapping, capsule networks seem to recognize them with good accuracy.

    Several research groups in astrophysics have started using deep neural networks, in particular convolutional neural network (CNN) to deal with astrophysical images (including recent work by a former graduate student of the PI [2]). However, no one has analyzed astrophysical images with Capsule Network yet. We aim to pioneer the use of capsule networks for astronomical research.
 
    In the area of astronomical observations, there are often cases where our target of interest is overlapping with other objects. It is often the case that a galaxy is overlapping another galaxy (an extreme example is shown in Fig.2 [3]) purely due to statistical chance. 

## Install

This project is developed for Python3.5 interpreter on linux machine. Using Anaconda virtual environment is recommended.

To install dependencies, simply run:

```pip install -r requirment.txt```

This project uses TensorFlow, a machine learning library developed and maintained by Google in principle.

We will be using tensorflow version 1.4.0 (a subtle difference has been observed in tensorflow 1.7.0, see network.py for details),

```pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.0-cp35-cp35m-linux_x86_64.whl```

users can choose to install its GPU optimized version accordingly,

```pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl```

To install cv2 in Anaconda (optional):

```conda install -c menpo opencv=2.4.11```

## Train

Simply run this in terminal:

```python main.py```

