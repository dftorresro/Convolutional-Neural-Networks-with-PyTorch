# Convolutional Neural Networks with PyTorch

## Project Overview

This project focuses on implementing a **Convolutional Neural Network (CNN)** for image recognition using the **PyTorch** deep learning framework. Two well-known datasets, **MNIST** (digit classification) and **CIFAR-10** (object recognition), are used to demonstrate CNNs in action. The project highlights the key components of building, training, and evaluating CNN models for image classification.

This project is implemented in a Jupyter notebook (`.ipynb` file), making it easy to follow and execute.

## Theoretical Background

### Convolutional Neural Networks (CNNs)

CNNs are a class of deep neural networks commonly used for processing grid-like data such as images. The architecture is composed of multiple layers, including convolutional layers that automatically learn filters for feature extraction, and fully connected layers that perform classification.

A basic CNN consists of the following key components:
- **Convolutional Layers**: These layers apply filters to the input images to detect patterns such as edges, textures, and more complex features.
- **Activation Functions**: Non-linear functions like ReLU are applied after convolution to introduce non-linearity, allowing the network to model more complex patterns.
- **Pooling Layers**: Max pooling layers downsample the spatial dimensions of the data, reducing its size and computational cost while retaining important features.
- **Fully Connected Layers**: After flattening the data, these layers perform the final classification.
- **Softmax Function**: This activation function is used in the output layer for multi-class classification, converting logits into probabilities for each class.

## Key Skills Demonstrated

- **Building a CNN from Scratch**: Implemented a simple CNN architecture using PyTorch, applying convolutional, activation, pooling, and fully connected layers.
- **Image Classification**: Trained and evaluated the CNN on both the MNIST and CIFAR-10 datasets, showcasing the model's ability to recognize handwritten digits and various objects.
- **PyTorch Environment**: Gained practical experience using PyTorch to define models, apply optimization techniques, and measure performance.
- **Softmax for Multi-Class Classification**: Used softmax activation in the output layer to handle the multi-class classification problem for both datasets.

## Project Structure

1. **Dataset Handling**:
   - Used the **MNIST** dataset, a collection of grayscale images of handwritten digits.
   - Applied the **CIFAR-10** dataset, containing 60,000 color images in 10 distinct classes.
   - Used PyTorch’s `torchvision` package to load and preprocess these datasets.

2. **CNN Architecture**:
   - **Convolutional Layers**: Two convolutional layers, each with 32 filters of size (3,3), stride (1,1), and zero-padding ('same') to preserve the spatial dimensions.
   - **Activation Functions**: Applied ReLU after each convolution to introduce non-linearity.
   - **Pooling Layer**: Max pooling with a (2,2) window and stride (2,2) to reduce the spatial dimensions by half.
   - **Flattening**: The output from the final pooling layer is flattened into a vector to feed into the fully connected layer.
   - **Fully Connected Layer**: A softmax function in the output layer for class prediction.

3. **Model Training**:
   - Trained the CNN using backpropagation with stochastic gradient descent.
   - Measured accuracy and loss during training and validation.
   - Applied data augmentation to improve model robustness, especially for the CIFAR-10 dataset.

4. **Evaluation**:
   - Evaluated the model performance on both test datasets, using accuracy as the primary metric.

## Notable Results

- **MNIST**: The CNN achieved high accuracy on the MNIST dataset, successfully recognizing handwritten digits.
- **CIFAR-10**: Achieved reasonable classification performance on CIFAR-10, given its higher complexity compared to MNIST.
- **Visualization**: Displayed training loss and accuracy, showing the model's learning progress over epochs. Visualized example predictions from the CNN to interpret its performance.

## Lessons Learned

- **Understanding CNNs**: Developed a solid understanding of how CNNs work, particularly how convolutional layers capture spatial hierarchies in images.
- **PyTorch Proficiency**: Gained hands-on experience in building, training, and evaluating deep learning models in PyTorch.
- **Dataset Differences**: Recognized the challenges of applying CNNs to more complex datasets like CIFAR-10 compared to simpler ones like MNIST.
