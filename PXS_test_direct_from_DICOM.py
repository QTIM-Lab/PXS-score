'''
Obtaining PXS scores directly from a DICOM input file. 
- there is an "autocrop" function which removes black boxes introduced in pre-processing for testing (some datasets with chest x-rays have black borders)
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

import cv2
import pydicom
from medpy.io.load import load

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

def autocrop(image, threshold=2):
    """Crops any edges below or equal to threshold
    Crops blank image to 1x1.
    Returns cropped image.
    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2
    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]
    return image

def img_processing(input_image):
    '''
    processes PIL image file -- with addition of crop function for black/near black border
    '''
    output_image = np.array(input_image)
    output_image = Image.fromarray(output_image)

    transf = transforms.Compose([
        transforms.Resize(336), 
        transforms.CenterCrop(320),
        transforms.ToTensor()
    ])

    output_image = transf(output_image)

    output_image = np.repeat(output_image, 3, 0)
    output_image= output_image[np.newaxis, ...]
    output_image = Variable(output_image).cuda()

    return output_image

def anchor_inference(img_anchor, image_path, net):
    '''
    takes img_anchor list object, image path of interest, and siamese network
    '''

    img_comparison = img_processing(Image.fromarray(image_path))

    save_euclidean_distance = []
    net.eval()
    for j in range(len(img_anchor)):
        output1, output2 = net.forward(img_anchor[j], img_comparison)
        euclidean_distance = F.pairwise_distance(output1, output2)
        save_euclidean_distance.append(euclidean_distance.item())

    # take median euclidean distance compared to the the pool of normals
    return statistics.median(save_euclidean_distance)

def dcm2img2jpg2pxs(dcm_file_path, img_anchor, net):
    """Extract the image from a path to a DICOM file."""
    # modified from Jeremy Irvin so as not to use pydicom-gdcm
    # Read the DICOM and extract the image.
    dcm_file = pydicom.dcmread(dcm_file_path, stop_before_pixels=True)

    curr_img, curr_header = load(dcm_file_path)
    raw_image = np.squeeze(curr_img).T.astype(np.float)

    assert len(raw_image.shape) == 2,\
        "Expecting single channel (grayscale) image."

    # # The DICOM standard specifies that you cannot make assumptions about
    # # unused bits in the representation of images, see Chapter 8, 8.1.1, Note 4:
    # # http://dicom.nema.org/medical/dicom/current/output/html/part05.html#chapter_8
    # # pydicom doesn’t exclude those unused bits by default, so we need to mask them
    # raw_image = np.bitwise_and(raw_image, (2 ** (dcm_file.HighBit + 1) -
    #                                        2 ** (dcm_file.HighBit -
    #                                              dcm_file.BitsStored + 1)))

    # Normalize pixels to be in [0, 255].
    raw_image = raw_image - raw_image.min()
    normalized_image = raw_image / raw_image.max()
    rescaled_image = (normalized_image * 255).astype(np.uint8)

    # Correct image inversion.
    if dcm_file.PhotometricInterpretation == "MONOCHROME1":
       rescaled_image = cv2.bitwise_not(rescaled_image)

    # Autocrop
    rescaled_image = autocrop(rescaled_image)

    # Perform histogram equalization
    adjusted_image = cv2.equalizeHist(rescaled_image)

    # jpeg encoding, as per CheXpert/MIMIC
    _, encimg = cv2.imencode('.jpg', adjusted_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) #encode the image using the same preprocessing as chexpert
    decimg = cv2.imdecode(encimg, 0)
    score = anchor_inference(img_anchor, decimg, net)

    return score

# anchor images from MGH training set -- used in updated PXS model in <link pending> 

image_dir = '/home/mgh_covid_training_set_anchors/'
anchor_table = pd.read_csv(image_dir + 'MGH_Covid_Training_Set_anchors.csv') # anchor_table contains paths to N normal chest x-rays
img_anchor = []
for a in tqdm(range(len(anchor_table))):
    image_path = image_dir + anchor_table['Admission_CXR_Accession'][a] + '.jpg'
    img_anchor.append(img_processing(Image.open(image_path)))

# # anchor images from CheXpert validation -- used in original PXS model in https://pubs.rsna.org/doi/10.1148/ryai.2020200079
# image_dir = '/home/Public_Datasets/chexpert/'
# anchor_table = pd.read_csv(working_path + 'anchor_table.csv')
# img_anchor = []
# for a in tqdm(range(len(anchor_table))):
#     image_path = image_dir + anchor_table.iloc[a].Path
#     img_anchor.append(img_processing(Image.open(image_path)))


# pickle the image anchors
pickle.dump(img_anchor, open(image_dir + "img_anchor.pickl", "wb"))
# load image anchors
img_anchor = pickle.load(open(image_dir + "img_anchor.pickl", "rb"))

# get test set annotations -- contains full file paths for DICOMs
annot_table = pd.read_csv(working_path + 'MGH_Covid_Test_Set_noID.csv')

PXS_score = []
for i in tqdm(range(len(annot_table))):
    acc = annot_table.iloc[i].CXR_Accession
    PXS = dcm2img2jpg2pxs(file_path, img_anchor, net)
    PXS_score.append(PXS)

annot_table['PXS_score'] = PXS_score
annot_table.to_csv(working_path + 'PXS_score_results.csv')

scipy.stats.pearsonr(annot_table['PXS_score'], annot_table['mRALE'])


'''
one image inference
'''
dcm2img2jpg2pxs(file_path, img_anchor, net)









