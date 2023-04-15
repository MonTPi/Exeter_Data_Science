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
for data_type in os.listdir(dataset_dir):
    os.mkdir(os.path.join('dataset', data_type))
    os.mkdir(os.path.join('dataset', data_type, 'train'))
    os.mkdir(os.path.join('dataset', data_type, 'val'))
    os.mkdir(os.path.join('dataset', data_type, 'test'))
    
    for cls in os.listdir(os.path.join(dataset_dir, data_type)):
    os.mkdir(os.path.join('dataset', data_type, 'train', cls))
    os.mkdir(os.path.join('dataset', data_type, 'val', cls))
    os.mkdir(os.path.join('dataset', data_type, 'test', cls))
    folder_imgs = os.listdir(os.path.join(dataset_dir, data_type, cls))
    for j in range(len(folder_imgs)):
        rand = np.random.random()
        if rand < 0.1:
            shutil.copyfile(os.path.join(os.path.join(dataset_dir, data_type, cls, folder_imgs[j])), 
                                            os.path.join('dataset', data_type, 'val', cls, folder_imgs[j]))
        elif rand < 0.2:
            shutil.copyfile(os.path.join(os.path.join(dataset_dir, data_type, cls, folder_imgs[j])), 
                                            os.path.join('dataset', data_type, 'test', cls, folder_imgs[j]))
        else:
            shutil.copyfile(os.path.join(os.path.join(dataset_dir, data_type, cls, folder_imgs[j])), 
                                            os.path.join('dataset', data_type, 'train', cls, folder_imgs[j]))
