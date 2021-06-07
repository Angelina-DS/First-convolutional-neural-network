# Introduction to convolutional neural network
We're going to train an artificial neural network on images of handwritten digits and check that the trained network is able to identify and classify new images that it did not see during the training. We will use the MNIST database, which contains 60.000 images (28x28 pixels) for training and 10.000 for testing. <br>

We'll use convolutional neural network because such network are able to learn filters that are applied to each image in order to highlight specific features, in particular which features of an image (curve, intensity, shape ...) are important for the classification. <br>

From the root of the project (First-convolutional-neural-network/): <br>
``` $ pip3 install -r requirements.txt ``` <br>
``` $ python3 main.py ``` <br>