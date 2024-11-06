import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras as kp
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Check TensorFlow and Keras versions
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


"""Step 1: Data processing"""
# breaking section into 4 sub-steps

# Sub-step 1: Checking Size of image

input_width, input_height, input_channel = 500,500, 3;
batch_size = 32;


# Sub-step 2: call the train and valid directory

train_data = r"C:\Users\gshar\OneDrive - Toronto Metropolitan University (RU)\Documents\GitHub\Gandhrav_Sharma_AER850_Project_2\Project 2 Data\Data\train"
valid_data = r"C:\Users\gshar\OneDrive - Toronto Metropolitan University (RU)\Documents\GitHub\Gandhrav_Sharma_AER850_Project_2\Project 2 Data\Data\valid"


# Sub-step 2: data augmentation

train_data_class = ImageDataGenerator(
    shear_range=0.2,
    zoom_range = 0.2,
    horizontal_flip= True,
    rotation_range = 40,
    )

valid_data_class = ImageDataGenerator(rescale = 1./255)


# Sub-step 3: train and validation generators
train_generator = train_data_class.flow_from_directory(
    train_data,
    target_size=(input_width, input_height),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = valid_data_class.flow_from_directory(
    valid_data,
    target_size=(input_width, input_height),
    batch_size=batch_size,
    class_mode='categorical'
)


""" 2.2 Step 2: Neural Network Architecture Design """

# start creating layers



# Sub-step 2.2.1 stacking all the layers for model 1

CNN_model = Sequential() # stacking the layer in sequence
CNN_model.add(Conv2D(32,(3,3), activation='relu',input_shape=(input_width, input_height, input_channel)))
CNN_model.add(MaxPooling2D(pool_size=(2, 2)))

# second stack
CNN_model.add(Conv2D(64,(3,3), activation='relu'))
CNN_model.add(MaxPooling2D(pool_size=(2, 2)))

# third stack
CNN_model.add(Conv2D(128, (3, 3), activation='relu'))  # try 'LeakyRelu' or 'elu'
CNN_model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten 
CNN_model.add(Flatten()),
CNN_model.add(Dense(64, activation = 'relu')), # fully connected layer
CNN_model.add(Dropout(0.5))
CNN_model.add(Dense(1, activation = 'sigmoid')) # sigmoid due to binary classification

print(CNN_model.summary())






