'''
Testing of PXS score
'''

# PyTorch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torch.autograd import Variable
  
# other modules 
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statistics
import pickle
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import scipy
from sklearn import metrics
 
# WORKING DIRECTORY
working_path = '/home/PXS_score/'
os.chdir(working_path)

from PXS_classes import SiameseNetwork_DenseNet121

# CUDA for PyTorch
os.environ['CUDA_VISIBLE_DEVICES']='0' # pick GPU to use
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
  
# loading model 
model_path = '/home/PXS_score/COVID_model/PXS_score_model.pth'

net = SiameseNetwork_DenseNet121().cuda()
net.load_state_dict(torch.load(model_path))

def img_processing(input_image):
    '''
    processes PIL image file
    '''
    
    transf = transforms.Compose([
        transforms.Resize(336), # maybe better to factor out resize step
        transforms.CenterCrop(320),
        transforms.ToTensor()
    ])

    output_image= transf(input_image)
    output_image = np.repeat(output_image, 3, 0)
    output_image= output_image[np.newaxis, ...]
    output_image = Variable(output_image).cuda()

    return output_image

def anchor_inference(img_anchor, image_path, net):
    '''
    takes img_anchor list object, image path of interest, and siamese network
    '''

    img_comparison = img_processing(Image.open(image_path))

    save_euclidean_distance = []
    net.eval()
    for j in range(len(img_anchor)):
        output1, output2 = net.forward(img_anchor[j], img_comparison)
        euclidean_distance = F.pairwise_distance(output1, output2)
        save_euclidean_distance.append(euclidean_distance.item())

    # take median euclidean distance compared to the the pool of normals
    return statistics.median(save_euclidean_distance)

# anchor images from CheXpert validation
image_dir = '/home/Public_Datasets/chexpert/'
anchor_table = pd.read_csv(working_path + 'anchor_table.csv') # anchor_table contains paths to N 'No Finding' CheXpert images
img_anchor = []
for a in tqdm(range(len(anchor_table))):
    image_path = image_dir + anchor_table.iloc[a].Path
    img_anchor.append(img_processing(Image.open(image_path)))

# get test set annotations
annot_table = pd.read_csv(working_path + 'MGH_Covid_Test_Set_noID.csv')

image_dir_mgh = '/home/mgh_covid_test_set_jpg/'
PXS_score = []
for i in tqdm(range(len(annot_table))):
    acc = annot_table.iloc[i].Admission_CXR_Accession
    PXS = anchor_inference(img_anchor, image_dir_mgh + acc + '.jpg', net)
    PXS_score.append(PXS)

annot_table['PXS_score'] = PXS_score
annot_table.to_csv(working_path + 'PXS_score_results.csv')

scipy.stats.spearmanr(annot_table['PXS_score'], annot_table['mRALE'])


# for inference for a single image, returns the median Euclidean distance for the image-of-interest at the image_path 
# relative to a pool of normal images (img_anchor), i.e. the PXS score
anchor_inference(img_anchor, image_path, net)


