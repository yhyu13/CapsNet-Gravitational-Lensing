# Predict Gravitational Lensing with Capusle Network

---

## Introduction
Capsule network (citation needed) is a new type of neural network proposed by Geoffrey Hinton. Hinton argues that the convolution neural network has several problems, such as they cannot handle the orientation information very well. With capsule network, the neurons are connected by “vector” instead of scalar weight. Therefore, capsule network can remember the “pose” if the object in images. For example, when digits are overlapping: capsule networks seems to recognize them with good accuracy.
 
In the area of astronomical observation, there are chances that our target of interested is actually overlapping with other objects, say a galaxy is overlapping another galaxy (https://apod.nasa.gov/apod/ap050507.html). The reason is that when we look into the sky with CCD telescopes, we are actually mapping the 3D distribution of objects into 2D pixelated images. 
 
Several research groups in astrophysics have started using deep neural networks, in particular convolutional neural network (CNN) to deal with astrophysical images. However, no one has analyzed astrophysical images with Capsule Network yet. Therefore, we would try to be the first group to utilize the CapsNet for research in our field.
 
We would start with images of strong gravitational lensing. According to Einstein’s General Relativity, light from distant galaxies would be bent when passing through foreground mass (e.g. galaxies, or galaxies cluster, dark matter halo...etc) and would form an arc like image on the sky. By studying gravitational lensing, we could tell the underlying dark matter distribution in the foreground galaxies, and hence helps up understanding the universe. 

Author: Joshua Yao-Yu Lin

## Install

This project is developed for Python3.5 interpreter on linux machine. Using Anaconda virtual environment is recommended.

To install dependencies, simply run:

```pip install -r requirment.txt```

To install cv2 in Anaconda:

```conda install -c menpo opencv=2.4.11```

## Train

Simply run this in terminal:

```python main.py```

