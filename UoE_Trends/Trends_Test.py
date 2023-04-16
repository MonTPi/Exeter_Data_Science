import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix
import cv2
import random
import os
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

dataset_dir = '/Users/jdlilley/Desktop/Data Science and Modelling/Python +/VisualStudio/Exeter_Data_Science/UoE_Trends/Multi Cancer/Kidney Cancer'

if not os.path.exists('dataset'):
    os.mkdir('dataset')
cancer_type = os.listdir(dataset_dir)
cancer_type = cancer_type[1:3]
for data_type in cancer_type:
    os.mkdir(os.path.join('dataset', data_type))
    os.mkdir(os.path.join('dataset', data_type, 'train'))
    os.mkdir(os.path.join('dataset', data_type, 'val'))
    os.mkdir(os.path.join('dataset', data_type, 'test'))
    folder_imgs = os.listdir(os.path.join(dataset_dir, data_type,))
    for j in range(len(folder_imgs)):
        rand = np.random.random()
        if rand < 0.1:
            shutil.copyfile(os.path.join(os.path.join(dataset_dir, data_type, folder_imgs[j])), 
                                            os.path.join('dataset', data_type, 'val', folder_imgs[j]))
        elif rand < 0.2:
            shutil.copyfile(os.path.join(os.path.join(dataset_dir, data_type, folder_imgs[j])), 
                                            os.path.join('dataset', data_type, 'test', folder_imgs[j]))
        else:
            shutil.copyfile(os.path.join(os.path.join(dataset_dir, data_type, folder_imgs[j])), 
                                            os.path.join('dataset', data_type, 'train', folder_imgs[j]))


data_dir = 'dataset'

datasets_dirs = os.listdir(data_dir)
print(datasets_dirs)
