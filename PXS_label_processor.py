'''
Process the CheXpert labels for input into PXS_train.py
'''

import os
import pandas as pd
from tqdm import tqdm

working_path = '/home/PXS_score/'
os.chdir(working_path)

 
### PROCESSING LABELS FOR CHEXPERT ####

# training

CheXpert_dir = '/home/Public_Datasets/chexpert/'

label_csv = pd.read_csv(CheXpert_dir + 'CheXpert-v1.0/train.csv')

# only AP images
updated_csv = label_csv[label_csv['AP/PA'] == 'AP']

# fill empty cells with 0
updated_csv = updated_csv.fillna(0)

# create "abnormal_lung" label, where if any lung abnormality label is present, abnormal lung label = 1
abnormal_lung = []
for i in tqdm(range(len(updated_csv))):

    # absolute value to deal with -1 uncertainty labels, which will be counted as positive +1 labels
    c0 = abs(updated_csv.iloc[i]['Lung Opacity'])     
    c1 = abs(updated_csv.iloc[i]['Lung Lesion']) 
    c2 = abs(updated_csv.iloc[i]['Edema'])
    c3 = abs(updated_csv.iloc[i]['Consolidation'])
    c4 = abs(updated_csv.iloc[i]['Pneumonia'])
    c5 = abs(updated_csv.iloc[i]['Atelectasis'])
    
    c6 = abs(updated_csv.iloc[i]['No Finding'])

    if sum([c0, c1, c2, c3, c4, c5]) >= 1:
        abnormal_lung.append(1)
    elif c6 == 1: # 'No Finding' label
        abnormal_lung.append(0)
    else: 
        abnormal_lung.append(None)

updated_csv['abnormal_lung'] = abnormal_lung

updated_csv.to_csv(working_path + 'chexpert_train_updated.csv')
 

# validation

CheXpert_dir = '/home/home/ken.chang/mnt/2015P002510/Public_Datasets/chexpert/'

label_csv = pd.read_csv(CheXpert_dir + 'CheXpert-v1.0/valid.csv')

# only AP images
updated_csv = label_csv[label_csv['AP/PA'] == 'AP']

# create "abnormal_lung" label, where if any lung abnormality label is present, abnormal lung label = 1
abnormal_lung = []
for i in tqdm(range(len(updated_csv))):

    # absolute value to deal with -1 uncertainty labels, which will be counted as positive +1 labels
    c0 = abs(updated_csv.iloc[i]['Lung Opacity'])     
    c1 = abs(updated_csv.iloc[i]['Lung Lesion']) 
    c2 = abs(updated_csv.iloc[i]['Edema'])
    c3 = abs(updated_csv.iloc[i]['Consolidation'])
    c4 = abs(updated_csv.iloc[i]['Pneumonia'])
    c5 = abs(updated_csv.iloc[i]['Atelectasis'])
    
    c6 = abs(updated_csv.iloc[i]['No Finding'])

    if sum([c0, c1, c2, c3, c4, c5]) >= 1:
        abnormal_lung.append(1)
    elif c6 == 1: # 'No Finding' label
        abnormal_lung.append(0)
    else: 
        abnormal_lung.append(None)

updated_csv['abnormal_lung'] = abnormal_lung

updated_csv.to_csv(working_path + 'chexpert_valid_updated.csv')



