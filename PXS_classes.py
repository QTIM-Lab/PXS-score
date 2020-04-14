''' 
Custom classes for PXS score, including siamese neural network

'''

# PyTorch modules
import torch
from torch import nn 
from torch.utils import data 
import torch.nn.functional as F 
from torchvision import transforms, models

# other modules
import os
import pandas as pd
import numpy as np
from PIL import Image
import random
import pydicom
import cv2
 

class CheXpert_Dataset(data.Dataset):
    """ 
    Create dataset representation of CheXpert data
    - This class returns image pairs with a change label (i.e. change vs no change in abnormal_lung) and other metadata
    - Image pairs are sampled so that there are an equal number of change vs no change labels
    - Epoch size can be set for empirical testing
  
    Concepts adapted from: 
    - https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """  
    def __init__(self, patient_table, image_dir, epoch_size, transform=None):
        """
        Args:
            patient_table (pd.dataframe): dataframe containing relative image paths, abnormal_lung, and other metadata
            image_dir (string): directory containing image files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.patient_table = patient_table
        self.image_dir = image_dir # note, images are already 8-bit, full-sized JPEGs
        self.transform = transform
        self.epoch_size = epoch_size 
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
 
    def __len__(self):
        return self.epoch_size
 
    def __getitem__(self, idx): 
        
        patient_table = self.patient_table
        # separate tables for abnormal lung labels 0 and 1
        table_A = patient_table[patient_table['abnormal_lung'] == 0]
        table_B = patient_table[patient_table['abnormal_lung'] == 1]

        # goal is 50:50 distribution of change vs no change
        change_binary = random.randint(0,1) 

        # if no change 
        if change_binary == 0:

            # select two random images 
            table_A_or_B = random.randint(0,1)

            if table_A_or_B == 0:
                pick0 = random.choice(range(len(table_A)))
                pick1 = random.choice(range(len(table_A)))
                img0 = Image.open(self.image_dir + table_A.iloc[pick0]['Path'])
                img1 = Image.open(self.image_dir + table_A.iloc[pick1]['Path'])

                abnormal_lung_0 = 0
                abnormal_lung_1 = 0

            if table_A_or_B == 1:
                pick0 = random.choice(range(len(table_B)))
                pick1 = random.choice(range(len(table_B)))
                img0 = Image.open(self.image_dir + table_B.iloc[pick0]['Path'])
                img1 = Image.open(self.image_dir + table_B.iloc[pick1]['Path'])

                abnormal_lung_0 = 1
                abnormal_lung_1 = 1

        # if change -- note: direction of change does not matter in this step
        if change_binary == 1: 

            pick0 = random.choice(range(len(table_A)))
            pick1 = random.choice(range(len(table_B)))
            img0 = Image.open(self.image_dir + table_A.iloc[pick0]['Path'])
            img1 = Image.open(self.image_dir + table_B.iloc[pick1]['Path'])

            abnormal_lung_0 = 0
            abnormal_lung_1 = 1

        # 0 for no change, 1 for change (difference between labels)
        if abnormal_lung_0 == abnormal_lung_1:
            change_label = 0
        else:
            change_label = 1
 
        meta = {"path0": patient_table.iloc[pick0]['Path'],
                "path1": patient_table.iloc[pick1]['Path'],
                "abnormal_lung_0": abnormal_lung_0,
                "abnormal_lung_1": abnormal_lung_1,
                }

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1) 

        return img0, img1, change_label, meta

 
class MGH_Dataset(data.Dataset):
    """ 
    Create dataset representation of MGH data
    - This class returns image pairs with a label for MSE loss (difference in Euclidean distance)
    - Epoch size can be set for empirical testing
  
    Concepts adapted from: 
    - https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """  
    def __init__(self, patient_table, epoch_size, transform=None):
        """
        Args:
            patient_table (pd.dataframe): dataframe containing relative image paths, abnormal_lung, and other metadata
            image_dir (string): directory containing image files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.patient_table = patient_table
        self.transform = transform
        self.epoch_size = epoch_size 
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
 
    def __len__(self):
        return self.epoch_size
 
    def __getitem__(self, idx): 
        
        table_A = self.patient_table

        # if no change 
        pick0 = random.choice(range(len(table_A)))
        pick1 = random.choice(range(len(table_A)))

        img0 = Image.open(table_A.iloc[pick0]['Path'])
        img1 = Image.open(table_A.iloc[pick1]['Path'])

        mrale0 = float(table_A.iloc[pick0]['mRALE'])
        mrale1 = float(table_A.iloc[pick1]['mRALE'])
 
        change_label = abs(mrale1 - mrale0)
 
        meta = {"path0": self.patient_table.iloc[pick0]['Path'],
                "path1": self.patient_table.iloc[pick1]['Path'],
                "mrale0": mrale0,
                "mrale1": mrale1,
                } 

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1) 

        return img0, img1, change_label, meta
        

class SiameseNetwork_DenseNet121(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    DenseNet 121 from Pytorch library
    """ 
    def __init__(self):
        super(SiameseNetwork_DenseNet121, self).__init__()
        self.cnn0 = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=True)
        # map last layer from 1000 node outcome to three nodes
        #self.cnn0.fc = nn.Linear(1000, 3) # mapping input image to a 3D space

    def forward_once(self, x):
        output = self.cnn0(x)
        return output

    def forward(self, input0, input1):
        output0 = self.forward_once(input0)
        output1 = self.forward_once(input1)
        return output0, output1
  
 
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    """ 

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output0, output1, label):
        euclidean_distance = F.pairwise_distance(output0, output1)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class MSELoss(torch.nn.Module):
    """
    MSE loss
    """ 

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output0, output1, label):
        euclidean_distance = F.pairwise_distance(output0, output1)
    
        return F.mse_loss(euclidean_distance,label)

