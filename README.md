# Image classification with CIFAR-10 dataset
Image Classification is a fundamental task that attempts to comprehend an entire image as a whole. The goal is to classify the image by assigning it to a specific label. Typically, Image Classification refers to images in which only one object appears and is analyzed. In contrast, object detection involves both classification and localization tasks, and is used to analyze more realistic cases in which multiple objects may exist in an image.
## About the dataset
The CIFAR-10 dataset is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.

![d](https://user-images.githubusercontent.com/68856803/89033560-a9573600-d354-11ea-9ff2-82ce285518b6.png)

# Flowchart

# Steps to implement this project:
We will implement this project by using Convolution neural network model. This model looks as following:

![e](https://user-images.githubusercontent.com/68856803/89042936-63569e00-d365-11ea-8170-04303e0108e9.png)




(a) Import the required packages and modules to create our CNN model

(b) Import the CIFAR-10 dataset with the help of keras

(c) Normalize the dataset and convert the pixel values of the dataset to float type

(d) Perform one-hot encoding for target classes

(e) Create the  CNN model and add a softmax activation function for output layer

(f) Compile the model with the help of 'sgd' optimizer (stochastic gradient descent)

(g) Train the model on 10 epochs and take the batch size as 32

(h) Finally make a dictionary to map to the output classes and make predictions from the model

