import h5py
import pandas as pd
from pandas import *
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import functional as F
from torchvision import datasets, models, transforms
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import random
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
import numpy
import os

from dataset import CreateDataset, normalize, denormalize

test_tran_dataset = CreateDataset(path= r"/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/Madhav/mtnn/Datasets/NitrUAVCorridorV1/Translation(Angle))/TestTranslation(Angle).h5", phase= 'test', mode= "tran")
test_rot_dataset = CreateDataset(path= r"/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/Madhav/mtnn/Datasets/NitrUAVCorridorV1/Rotation(Distance)/TestRotation(Distance).h5", phase= 'test', mode="rot")

test_tran_loader = data.DataLoader(test_tran_dataset, batch_size=1)
test_rot_loader = data.DataLoader(test_rot_dataset, batch_size=1)

train_tran_dataset = CreateDataset(path= r"/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/Madhav/mtnn/Datasets/NitrUAVCorridorV1/Translation(Angle))/TrainTranslation(Angle).h5", phase= 'train', mode="tran")
train_rot_dataset = CreateDataset(path= r"/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/Madhav/mtnn/Datasets/NitrUAVCorridorV1/Rotation(Distance)/TrainRotation(Distance).h5", phase= 'train', mode="rot")

train_tran_loader = data.DataLoader(train_tran_dataset, batch_size=20)
train_rot_loader = data.DataLoader(train_rot_dataset, batch_size=20)

densenet_model = models.densenet161(pretrained=True)
num_features = densenet_model.classifier.in_features

task1 = nn.Linear(num_features, 1)
task2 = nn.Linear(num_features, 1)

densenet_model.classifier.add_module('task1', task1)
densenet_model.classifier.add_module('task2', task2)

criterion = nn.L1Loss()
optimizer = optim.Adam(densenet_model.classifier.parameters(), lr=0.0001)

def train(epochs):
    all_training_loss = []
    epochs_list = []
    if torch.cuda.is_available():
        densenet_model.cuda()
    else:
        densenet_model.cpu()
    for epoch in range(epochs):
        epochs_list.append(epoch)
        print(f"Epoch-{epoch+1} | \t", end="")
        training_loss = 0
        
        for (image1, label1), (image2, label2) in zip(train_tran_loader, train_rot_loader):
            
            if torch.cuda.is_available():
                image1, image2, label1, label2 = image1.cuda(), image2.cuda(), label1.cuda(), label2.cuda()
            
            image1 = image1.clone().detach()
            image2 = image2.clone().detach()
            label2 = normalize(label2)
            
            optimizer.zero_grad()
            
            features1 = densenet_model.features(image1.float())
            features1 = nn.functional.relu(features1, inplace=True)
            features1 = nn.functional.adaptive_avg_pool2d(features1, (1, 1))
            features1 = torch.flatten(features1, 1)
            output_1 = task1(features1)
            
            features2 = densenet_model.features(image2.float())
            features2 = nn.functional.relu(features2, inplace=True)
            features2 = nn.functional.adaptive_avg_pool2d(features2, (1, 1))
            features2 = torch.flatten(features2, 1)
            output_2 = task2(features2)
            
            loss1 = criterion(output_1.view(-1), label1.view(-1))
            loss2 = criterion(output_2.view(-1), label2.view(-1))
            loss = loss1 + loss2
            loss.backward()
            
            training_loss += loss.item()
            optimizer.step()
        else:
            densenet_model.eval()
            
            total_train_length = len(train_tran_loader) + len(train_rot_loader)
                    
            some_log = f"Training loss: {training_loss/total_train_length:.4f}"
            print(some_log)
            
            all_training_loss.append(training_loss/total_train_length)
            
            densenet_model.train()

        os.makedirs("Output", exist_ok=True)
        
        torch.save(densenet_model.state_dict(),os.path.join("Output", f"densenet_{epoch}.pth"))
        print(f"Model saved as densenet_{epoch}.pth")
        
def training_plot():
    # Loss values and epochs
    loss_values = []
    epochs = list(range(1, len(loss_values) + 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, linestyle='-')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    # plt.grid(True)
    plt.show()

def test():
    import torch.nn as nn
    import torchvision.models as models

    densenet_model = models.densenet161(pretrained=True)
    num_features = densenet_model.classifier.in_features

    task1 = nn.Linear(num_features, 1)
    task2 = nn.Linear(num_features, 1)

    densenet_model.classifier.add_module('task1', task1)
    densenet_model.classifier.add_module('task2', task2)

    checkpoint = torch.load("/content/drive/MyDrive/densenet_99.pth", map_location=torch.device('cpu'))
    densenet_model.load_state_dict(checkpoint)
    #Testing
    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    densenet_model.eval()
    cuda = torch.cuda.is_available()

    if cuda:
        densenet_model.cuda()
    else:
        densenet_model.cpu()

    angle_tolerance_deg = 5
    rotation_tolerance = angle_tolerance_deg * (3.14159 / 180)
    distance_tolerance = 0.15

    dist_correct = 0
    rot_correct = 0

    tran_loss1 = []
    tran_loss2 = []
    rot_loss1 = []
    rot_loss2 = []

    for image1, label1 in test_tran_loader:

        with torch.no_grad():
                if cuda:
                    image1, label1= image1.cuda(), label1.cuda()

                image1 = image1.float()
                features1 = densenet_model.features(image1.float())
                features1 = nn.functional.relu(features1, inplace=True)
                features1 = nn.functional.adaptive_avg_pool2d(features1, (1, 1))
                features1 = torch.flatten(features1, 1)

                output1 = task1(features1)

                loss1 = criterion1(output1.view(-1), label1.view(-1))
                loss2 = criterion2(output1.view(-1), label1.view(-1))

                dist_mae += loss1.item()
                dist_mse += loss2.item()

                output1 = denormalize(output1)
                label1 = denormalize(label1)

                tran_loss1.append(loss1.item())
                tran_loss2.append(loss2.item())

                dist_diff = torch.abs(output1 - label1)
                dist_correct += torch.sum(dist_diff <= distance_tolerance).item()

        for image2, label2 in test_rot_loader:
            label2 = normalize(label2)
            with torch.no_grad():
                if cuda:
                    image2, label2 = image2.cuda(), label2.cuda()

                image2 = image2.float()
                features2 = densenet_model.features(image2.float())
                features2 = nn.functional.relu(features2, inplace=True)
                features2 = nn.functional.adaptive_avg_pool2d(features2, (1, 1))
                features2 = torch.flatten(features2, 1)

                output2 = task2(features2)

                loss3 = criterion1(output2.view(-1), label2.view(-1))
                loss4 = criterion2(output2.view(-1), label2.view(-1))

                ang_mae += loss3.item()
                ang_mse += loss4.item()

                output2 = denormalize(output2)
                label2 = denormalize(label2)

                rot_loss1.append(loss3.item())
                rot_loss2.append(loss4.item())

                rot_diff = torch.abs(output2 - label2)
                rot_correct += torch.sum(rot_diff <= rotation_tolerance).item()

    # Calculate accuracy
    dist_accuracy = dist_correct / len(test_tran_loader)
    rot_accuracy = rot_correct / len(test_rot_loader)

    total_correct = dist_correct + rot_correct
    total_accuracy = total_correct / total_len

    # Display accuracy
    print("\nDistance Accuracy:", dist_accuracy)
    print("Rotation Accuracy:", rot_accuracy)
    print("Total Accuracy:", total_accuracy)

def training_plot(training_loss):

    epochs = list(range(1, len(tran_loss1) + 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, tran_loss1, linestyle='-')
    plt.plot(epochs, tran_loss2)
    plt.title('Testing Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Testing Loss')
    plt.legend(True)
    # plt.grid(True)
    plt.show()

    plt.clf()

    epochs = list(range(1, len(rot_loss1) + 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rot_loss1, linestyle='-')
    plt.plot(epochs, rot_loss2)
    plt.title('Testing Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Testing Loss')
    plt.legend(True)
    # plt.grid(True)
    plt.show()
    
def __main__():
    training_loss = train(100)
    training_plot(training_loss)
    test()