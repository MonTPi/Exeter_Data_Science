import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# Define image sizes and batch size
img_width, img_height = 512, 512
batch_size = 32

# Define directories
train_dir = '/Users/jdlilley/Desktop/Data Science and Modelling/Python +/VisualStudio/Exeter_Data_Science/UoE_Trends/Data_set_OGCNN/Training'
test_dir = '/Users/jdlilley/Desktop/Data Science and Modelling/Python +/VisualStudio/Exeter_Data_Science/UoE_Trends/Data_set_OGCNN/Testing' 

# Define data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit(train_generator,
                    epochs=10,
                    batch_size=batch_size,
                    validation_data=test_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, batch_size=batch_size)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Predict the classes of the test set
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.2).astype(int)

# Plot the confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

true_classes = test_generator.classes
class_names = test_generator.class_indices.keys()
cm = confusion_matrix(true_classes, predicted_classes)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(cm, cmap=plt.cm.Blues)
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion matrix')
plt.show()

import pandas as pd

# Create a DataFrame from the true and predicted classes
results_df = pd.DataFrame({'true_classes': true_classes, 'predicted_classes': predicted_classes.reshape(-1)})

# Save the DataFrame to a CSV file
results_df.to_csv('results_OG.csv', index=False)
