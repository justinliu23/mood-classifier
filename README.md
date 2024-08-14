# mood-classifier
 

## Installation Instructions

### Prerequisites
Before running this project, ensure you have the following software installed on your local machine:
- Python 3.6 or later
- Jupyter Notebook or JupyterLab
- pip (Python package installer)

### Dependencies
This project requires several Python libraries, which can be installed via pip. The main dependencies include:
- `numpy`: For numerical computations
- `matplotlib`: For plotting and visualizations
- `h5py`: For handling HDF5 file format
- `scipy`: For scientific computing tasks
- `PIL` (Pillow): For image processing
- `pandas`: For data manipulation and analysis
- `tensorflow`: For building and training deep learning models

## Features
- **Sequential Model**: A straightforward model built using Keras' Sequential API, designed for binary classification.
- **Functional API Model**: A more flexible model built using Keras' Functional API, capable of handling complex model architectures.
- **Training Visualization**: The project includes code for visualizing the training process, including loss and accuracy over epochs.

## Configuration
The models are configurable through parameters specified in the functions that define them. Key configurations include:
- **Input Shape**: The shape of the input images can be adjusted in the model definition functions.
- **Number of Filters**: The number of filters in convolutional layers can be modified.
- **Kernel Size**: The size of the convolutional kernels is configurable.
- **Activation Functions**: The choice of activation functions (e.g., ReLU, Sigmoid) can be tailored to the specific task.

## Project Structure
- **convolution_model_application.ipynb**: The main script containing the implementation of the models, data loading, training, and evaluation.
- **cnn_utils.py**: A utility script that contains helper functions for loading datasets and preprocessing.

## Model Architectures

### Sequential Model
The Sequential model implements a basic CNN architecture for binary classification. The architecture consists of:
- **ZeroPadding2D**
- **Conv2D**
- **BatchNormalization**
- **ReLU Activation**
- **MaxPooling2D**
- **Flatten**
- **Dense Output Layer with Sigmoid Activation**

This model is trained on the Happy House dataset to classify images as smiling or not smiling.

### Functional API Model
The Functional API model provides a more flexible architecture, capable of handling multiple inputs and outputs. The architecture includes:
- **Conv2D**
- **ReLU Activation**
- **MaxPooling2D**
- **Flatten**
- **Dense Output Layer with Softmax Activation**

This model is trained on the SIGNS dataset to classify hand signs representing numbers from 0 to 5.
