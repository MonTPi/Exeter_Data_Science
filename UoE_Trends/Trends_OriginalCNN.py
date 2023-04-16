import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Set the directories for the data
tumor_dir = '/Users/jdlilley/Desktop/Data Science and Modelling/Python +/VisualStudio/Exeter_Data_Science/UoE_Trends/Multi Cancer_Kidney/Kidney Cancer/kidney_normal'
non_tumor_dir = '/Users/jdlilley/Desktop/Data Science and Modelling/Python +/VisualStudio/Exeter_Data_Science/UoE_Trends/Multi Cancer_Kidney/Kidney Cancer/kidney_tumor'

# Load the tumor images
tumor_data = []
for filename in os.listdir(tumor_dir):
    img = load_img(os.path.join(tumor_dir, filename), target_size=(150, 150))
    img = img_to_array(img)
    tumor_data.append(img)
tumor_data = np.array(tumor_data)

# Load the non-tumor images
non_tumor_data = []
for filename in os.listdir(non_tumor_dir):
    img = load_img(os.path.join(non_tumor_dir, filename), target_size=(150, 150))
    img = img_to_array(img)
    non_tumor_data.append(img)
non_tumor_data = np.array(non_tumor_data)

# Create labels for the data
tumor_labels = np.ones(len(tumor_data))
non_tumor_labels = np.zeros(len(non_tumor_data))

# Combine the data and labels
data = np.concatenate((tumor_data, non_tumor_data), axis=0)
labels = np.concatenate((tumor_labels, non_tumor_labels), axis=0)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create data generator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Train the model
history = model.fit(datagen.flow(train_data, train_labels, batch_size=32), epochs=50, validation_data=(test_data, test_labels))

# Make predictions on the test set
y_pred = model.predict(test_data)
y_pred = (y_pred > 0.5)

# Compute the confusion matrix
cm = confusion_matrix(test_labels, y_pred)

# Print the confusion matrix
print(cm)

# Plot the accuracy and loss curves during training
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
