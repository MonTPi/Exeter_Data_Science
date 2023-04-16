import os
import random
import shutil

# Set seed to ensure consistent results
random.seed(42)

# Define directories for original data and new split data
original_dir = '/Users/jdlilley/Desktop/Data Science and Modelling/Python +/VisualStudio/Exeter_Data_Science/UoE_Trends/Multi Cancer_Kidney/Kidney Cancer'
train_dir = '/Users/jdlilley/Desktop/Data Science and Modelling/Python +/VisualStudio/Exeter_Data_Science/UoE_Trends/Data_set_OGCNN/Training'
test_dir = '/Users/jdlilley/Desktop/Data Science and Modelling/Python +/VisualStudio/Exeter_Data_Science/UoE_Trends/Data_set_OGCNN/Testing'

# Define classes and create subdirectories in train and test directories
classes = ["kidney_tumor", "kidney_normal"]
for directory in [train_dir, test_dir]:
    for class_name in classes:
        os.makedirs(os.path.join(directory, class_name), exist_ok=True)

# Define train/test split ratio
split_ratio = 0.8

# Loop through each class and split images into train/test directories
for class_name in classes:
    # Get all file names for this class
    filenames = os.listdir(os.path.join(original_dir, class_name))
    # Shuffle filenames randomly
    random.shuffle(filenames)
    # Split into train/test filenames
    split_index = int(len(filenames) * split_ratio)
    train_filenames = filenames[:split_index]
    test_filenames = filenames[split_index:]
    # Copy files into appropriate train/test subdirectories
    for filename in train_filenames:
        src_path = os.path.join(original_dir, class_name, filename)
        dst_path = os.path.join(train_dir, class_name, filename)
        shutil.copy(src_path, dst_path)
    for filename in test_filenames:
        src_path = os.path.join(original_dir, class_name, filename)
        dst_path = os.path.join(test_dir, class_name, filename)
        shutil.copy(src_path, dst_path)