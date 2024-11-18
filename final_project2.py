import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras as kp
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras.optimizers import Adam






# step 2
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU

from tensorflow.keras.optimizers import SGD
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)

# Check TensorFlow and Keras versions
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")


"""Step 1: Data processing"""
# breaking section into 4 sub-steps

# Sub-step 1: Checking Size of image

input_width, input_height, input_channel = 100,100, 3;
batch_size = 32;


# Sub-step 2: call the train and valid directory

train_data = r"C:\Users\gshar\OneDrive - Toronto Metropolitan University (RU)\Documents\GitHub\Gandhrav_Sharma_AER850_Project_2\Project 2 Data\Data\train"
valid_data = r"C:\Users\gshar\OneDrive - Toronto Metropolitan University (RU)\Documents\GitHub\Gandhrav_Sharma_AER850_Project_2\Project 2 Data\Data\valid"


# Sub-step 3: Data augmentation for training data
# This applies random transformations to the training data to improve generalization and prevent overfitting
train_data_class = ImageDataGenerator(
    shear_range=0.2,       # Randomly applies shearing transformations
    zoom_range=0.2,        # Randomly zooms into images
    horizontal_flip=True,  # Randomly flips images horizontally
    rotation_range=40      # Rotates images randomly within a range of 40 degrees
)


valid_data_class = ImageDataGenerator(rescale = 1./255)


train_generator = train_data_class.flow_from_directory(
    train_data,  # Path to training data
    target_size=(input_width, input_height),  # Resizing all images to 100x100
    batch_size=batch_size,  # Number of images per batch
    class_mode='categorical'  # Specifies that labels are one-hot encoded
)

valid_generator = valid_data_class.flow_from_directory(
    valid_data,  # Path to validation data
    target_size=(input_width, input_height),  # Resizing all images to 100x100
    batch_size=batch_size,  # Number of images per batch
    class_mode='categorical'  # Specifies that labels are one-hot encoded
)




# """ 2.3 Step 3: Hyperparameter Analysis """

# # start creating layers



# # #Sub-step 2.3.1 stacking all the layers for model 1

CNN_model = Sequential() # stacking the layer in sequence
CNN_model.add(Conv2D(128,(3,3), strides=(1, 1),activation='relu',input_shape=(input_width, input_height, input_channel)))
CNN_model.add(MaxPooling2D(pool_size=(2, 2)))

# second stack
CNN_model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
CNN_model.add(MaxPooling2D(pool_size=(2, 2)))

# third stack
CNN_model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
CNN_model.add(MaxPooling2D(pool_size=(2, 2)))

#fourth stack 
CNN_model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
CNN_model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten 
CNN_model.add(Flatten()),
CNN_model.add(Dense(256, activation = 'relu')), # fully connected layer # 128 neurons
CNN_model.add(Dense(32, activation = 'relu'))
CNN_model.add(Dropout(0.1))
CNN_model.add(Dense(3, activation = 'softmax')) 

print(CNN_model.summary())

# compile by adding backpropogation algorithm
#CNN_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) # 

################################################################ new one try this first b4 deleting 
learning_rate = 1e-5
CNN_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])
#######################################################3

"""2.4 Step 4: Model Evaluation """

history1 = CNN_model.fit(x= train_generator, validation_data = valid_generator, epochs=15,batch_size = 128, validation_split = 0.001)




## Begin Plotting the validation and train


## Plotting accuracy over epochs for Model 1
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'], label='Training Accuracy')
plt.plot(history1.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy for Model 1')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plotting Loss over epochs for Model 1 
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history1.history['loss'], label='Training Loss')  # Changed 'Loss' to 'loss'
plt.plot(history1.history['val_loss'], label='Validation Loss')  # Changed 'val_Loss' to 'val_loss'
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Model 1')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()





# """"Old Model """

# '''Sub-step 2.3.2 stacking all the layers for model 2'''

#CNN Model using LeakyReLU
# CNN_model2 = Sequential()  # stacking the layers in sequence
# CNN_model2.add(Conv2D(256, (3, 3), strides=(1, 1),input_shape=(input_width, input_height, input_channel)))
# CNN_model2.add(LeakyReLU(alpha=0.1))  # Adding LeakyReLU activation with alpha parameter
# CNN_model2.add(MaxPooling2D(pool_size=(2, 2)))

# second stack
# CNN_model2.add(Conv2D(256, (3, 3),strides=(1, 1)))
# CNN_model2.add(LeakyReLU(alpha=0.1))
# CNN_model2.add(MaxPooling2D(pool_size=(2, 2)))

# third stack
# CNN_model2.add(Conv2D(512, (3, 3),strides=(1, 1)))
# CNN_model2.add(LeakyReLU(alpha=0.1))  
# CNN_model2.add(MaxPooling2D(pool_size=(2, 2)))

# fourth stack
# CNN_model2.add(Conv2D(512, (3, 3),strides=(1, 1)))
# CNN_model2.add(LeakyReLU(alpha=0.1))  
# CNN_model2.add(MaxPooling2D(pool_size=(2, 2)))


# Flatten 
# CNN_model2.add(Flatten())
# CNN_model2.add(Dense(256))
# CNN_model2.add(LeakyReLU(alpha=0.1))
# CNN_model2.add(Dropout(0.2))

# Output Layer
# CNN_model2.add(Dense(3, activation='softmax'))  

# Print model summary
# print(CNN_model2.summary())

# Compile by adding backpropagation algorithm
# CNN_model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# history2 = CNN_model2.fit(x= train_generator, validation_data = valid_generator, epochs=10, batch_size=256,  verbose=1)


# Plotting accuracy over epochs for Model 2
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history2.history['accuracy'], label='Training Accuracy')
# plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Training and Validation Accuracy for Model 2')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# Plotting Loss over epochs for Model 2
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history2.history['loss'], label='Training Loss')
# plt.plot(history2.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss for Model 2')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()



""""Saving the model"""

# Save the models
CNN_model.save('model1.h5')
# model2.save('model2.h5')