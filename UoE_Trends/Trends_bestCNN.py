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

dataset_dir = '/Users/jdlilley/Desktop/Data Science and Modelling/Python +/VisualStudio/Exeter_Data_Science/UoE_Trends/Multi Cancer'

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

data_dir = 'dataset'

datasets_dirs = os.listdir(data_dir)

def load_data(dataset):
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, dataset, 'train'), 
                                         transform=data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, dataset, 'val'), 
                                       transform=data_transforms)    
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, dataset, 'test'), 
                                        transform=data_transforms)    
    
    return train_dataset, val_dataset, test_dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

def build_model(data):
    model = models.resnet18(pretrained=True)
    
    for parameter in model.parameters():
        parameter.required_grad = False

    model.fc = nn.Linear(in_features=512, out_features=len(data.classes))
    
    return model 

criterion = nn.CrossEntropyLoss()
criterion.to(device)

def validation(model, val_loader):
    
    val_loss = 0
    accuracy = 0
    
    for val_step, (images, labels) in enumerate(val_loader):
        images, labels = images.to(device), labels.to(device)
        
        preds = model(images)
        val_loss += criterion(preds, labels).item()
        
        accuracy += (preds.argmax(1) == labels).type(torch.float).sum().item()
        
    return val_loss, accuracy/len(val_loader.dataset)

def train(model, train_loader, val_loader, optimizer, epochs=4):
    
    model.to(device)
    
    for i in range(epochs):
        print("="*20)
        print(f"Epoch: {i+1}/{epochs}")
        
        model.train()
        
        training_loss = 0
        for train_step, (images, labels) in enumerate(train_loader):
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()

            if train_step % 50 == 0:
                
                model.eval()
                with torch.no_grad():
                    validation_loss, val_accuracy = validation(model, val_loader)
                
                print(f"Step: {train_step}/{len(train_loader)} \n\t"
                      f"Validation Loss: {validation_loss:.3f} |",
                      f"Validation Accuracy: {val_accuracy:.3f}")
                
                training_loss = 0
                model.train()


def test(model, test_loader):
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        accuracy = 0
        
        for images, labels in iter(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            pred = model(images)
            accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()
            
    print(f"Test Accuracy: {accuracy/len(test_loader.dataset):.3f}") 

def save_checkpoint(model, name, train_data):
    if not os.path.exists('models'):
        os.mkdir('models')
    
    model.class_to_idx = train_data.class_to_idx
    model = model.to('cpu')
    
    torch.save(model, f'models/{name}.pth')
    print(f'Saved to models/{name}.pth')

def load_model(filepath):
    model = torch.load(filepath)
    return model

def process_image(image_path):
    image = Image.open(image_path)
    
    if image.size[0] > image.size[1]:
        image.thumbnail((5000, 256))
    else:
        image.thumbnail((256, 5000))        
    
    np_img = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std
    
    np_img = np_img.transpose((2, 0, 1))
    
    return np_img

def predict(image_path, model):
    
    image = process_image(image_path)
    
    image = torch.from_numpy(image).type(torch.FloatTensor) #.to(device)
    image = image.unsqueeze(0)
    
    preds = model(image)
    probabilities = torch.exp(preds)
    
    top_probs, top_indices = probabilities.topk(len(model.class_to_idx))
    
    top_probs = top_probs.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]
    
    idx_to_classes = {value: key for key, value in model.class_to_idx.items()}
    
    top_classes = [idx_to_classes[index] for index in top_indices]
    
    return top_probs, top_classes

    
def show_results(image_path, model):
    
    probs, classes = predict(image_path, model)
    
    plt.figure(figsize=(12, 8))
    axs = plt.subplot(1, 2, 1)
    
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'{classes[0]}')
    
    plt.subplot(1, 2, 2)
    
    sb.barplot(x=classes, y=probs)
    plt.title('Probablities')
    plt.show()

def predict_chosen_img(folder, class_name, model):
    path = os.path.join("Multi Cancer", folder, class_name)
    chosen_img = os.listdir(path)[random.randint(0, len(os.listdir(path)))]
    
    show_results(os.path.join(path, chosen_img), model)

def show_confusion_matrix(test_loader, model):
    y_true = []
    y_pred = []
    model = model.to(device)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        preds = model(images)
        
        preds = (torch.max(torch.exp(preds), 1)[1]).data.cpu().numpy()
        y_pred.extend(preds)
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)
        
    classes = list(model.class_to_idx.keys())
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1), index=classes, columns=classes)
    sb.heatmap(df_cm, annot=True, cbar=False)
    plt.show()


                                                              
datasets_dirs[6]

training_data, val_data, test_data = load_data(datasets_dirs[6])

train_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

len(training_data), len(val_data), len(test_data)

kidney_cancer_labels = training_data.classes
kidney_cancer_labels

model7 = build_model(training_data)

optimizer7 = optim.Adam(model7.fc.parameters(), lr=0.001)

train(model7, train_loader, val_loader, optimizer7)

test(model7, test_loader)

save_checkpoint(model7, 'kidney_cancer', training_data)

model7_loaded = load_model('models/kidney_cancer.pth')

predict_chosen_img(datasets_dirs[6], kidney_cancer_labels[0], model7_loaded)

predict_chosen_img(datasets_dirs[6], kidney_cancer_labels[1], model7_loaded)

show_confusion_matrix(test_loader, model7_loaded)