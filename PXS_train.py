'''
Siamese neural network implementation for CXRs

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

# WORKING DIRECTORY
working_path = '/home/home/ken.chang/mnt/2015P002510/Matt/PXS_score/'
os.chdir(working_path)

# custom classes
from PXS_classes import CheXpert_Dataset, SiameseNetwork_DenseNet121, ContrastiveLoss

# CUDA for PyTorch
os.environ['CUDA_VISIBLE_DEVICES']='0' # pick GPU to use
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

''' 
Load imaging data with annotations
'''
 
# annotations
training_table = pd.read_csv(working_path + 'chexpert_train_updated.csv')
validation_table = pd.read_csv(working_path + 'chexpert_valid_updated.csv')

# processed image directory
image_dir = '/home/home/ken.chang/mnt/2015P002510/Public_Datasets/chexpert/'

# TRAINING DATA 

training_transforms = transforms.Compose([
    transforms.Resize(336), # maybe better to factor out resize step
    transforms.CenterCrop(320),
    transforms.ToTensor()
])

training_siamese_dataset = CheXpert_Dataset(patient_table = training_table, 
                                        image_dir = image_dir, 
                                        epoch_size = 6400,
                                        transform = training_transforms)
 
training_dataloader = torch.utils.data.DataLoader(training_siamese_dataset, 
                                                  batch_size=8, 
                                                  shuffle=False, 
                                                  num_workers=4)
  
# VALIDATION DATA 

validation_transforms = transforms.Compose([
    transforms.Resize(336), # maybe better to factor out resize step
    transforms.CenterCrop(320),
    transforms.ToTensor()
])

# note dataset v5 for inter-patient 50:50 comparisons
validation_siamese_dataset = CheXpert_Dataset(patient_table = validation_table, 
                                        image_dir = image_dir, 
                                        epoch_size = 200,
                                        transform = validation_transforms)
 
validation_dataloader = torch.utils.data.DataLoader(validation_siamese_dataset, 
                                                  batch_size=8, 
                                                  shuffle=False, 
                                                  num_workers=4)

'''
Training the siamese network 
Based on https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
Early stopping with saving of model by validation loss based on https://towardsdatascience.com/transfer-learning-with-convolutional-neural-networks-in-pytorch-dd09190245ce
'''
 
def siamese_training(training_dataloader, validation_dataloader, working_path, output_folder_name, learning_rate = 0.00002):
    '''
    - Implements siamese network training/validation with return of model
    - Implementation uses early stopping, saving the model with the best validation loss
 
    Arguments
    - training and validation dataloader objects
    - output_folder_name: directory to save model and annotations
    - learning_rate: for Adam optimizer
    ''' 
    net = SiameseNetwork_DenseNet121().cuda()
    criterion = ContrastiveLoss()
    criterion.margin = 50.0  # contrastive loss function margin
    optimizer = optim.Adam(net.parameters(),lr = learning_rate)
 
    # Initialization
    num_epochs = 1000
    training_losses, validation_losses = [], []

    # Early stopping initialization
    epochs_no_improve = 0
    max_epochs_stop = 3 # "patience" - number of epochs with no improvement in validation loss after which training stops
    validation_loss_min = np.Inf
    history = []

    output_dir = working_path + output_folder_name
    os.mkdir(output_dir)
    f = open(output_dir + "/history_training.txt", 'a+')
    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" + "Training starting now...\n")
    f.close()

    for epoch in range(0, num_epochs):
        print("epoch training started...")

        # keep track of training and validation loss each epoch
        training_loss = 0
        validation_loss = 0

        # keep track of euclidean_distance and label history each epoch
        training_euclidean_distance_history = []
        training_label_history = []
        validation_euclidean_distance_history = []
        validation_label_history = []

        # model set to train
        net.train()
          
        # training loop
        for i, data in tqdm(enumerate(training_dataloader, 0)):
            # train neural network
            img0, img1, label, meta = data
            img0 = np.repeat(img0, 3, 1) # repeat grayscale image in 3 channels (DenseNet requires 3-channel input) 
            img1 = np.repeat(img1, 3, 1)
            img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()  # send tensors to the GPU
            optimizer.zero_grad() # clear gradients
            output0, output1 = net.forward(img0, img1)
            loss_contrastive = criterion(output0, output1, label.float())
            loss_contrastive.backward()
            optimizer.step()

            # keep track of training loss
            training_loss += loss_contrastive.item()

            # save euclidean distance and label history 
            net.eval()
            output0, output1 = net.forward(img0, img1)
            net.train()
            euclidean_distance = F.pairwise_distance(output0, output1)
            euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu())
            training_euclidean_distance_history.extend(euclid_tmp)

            label_tmp = torch.Tensor.numpy(label.cpu())
            training_label_history.extend(label_tmp)

            print("training loop " + str(i) + " completed")

        else:
            print("validation started...")
            #turn off gradients for validation
            with torch.no_grad(): 
                net.eval() # set evaluation mode

                # determine validation loss
                for j, data2 in tqdm(enumerate(validation_dataloader, 0)):
                    img0, img1, label, meta = data2
                    img0 = np.repeat(img0, 3, 1) # repeat grayscale image in 3 channels (DenseNet requires 3-channel input) 
                    img1 = np.repeat(img1, 3, 1)
                    img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
                    output0, output1 = net.forward(img0, img1)
                    loss_contrastive = criterion(output0, output1, label.float())
                    validation_loss += loss_contrastive.item()
                     
                    # save euclidean distance and label history 
                    euclidean_distance = F.pairwise_distance(output0, output1)                
                    euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu())
                    validation_euclidean_distance_history.extend(euclid_tmp)

                    label_tmp = torch.Tensor.numpy(label.cpu())
                    validation_label_history.extend(label_tmp)

                    print("validation loop " + str(j) + " completed")

            # calculate average training and validation losses (averaged across batches for the epoch)
            training_loss_avg = training_loss/len(training_dataloader)
            validation_loss_avg = validation_loss/len(validation_dataloader)

            # Save the model if validation loss decreases
            if validation_loss_avg < validation_loss_min:
                # save model
                torch.save(net.state_dict(), output_dir + "/PXS_score_model.pth")
                # track improvement
                epochs_no_improve = 0
                validation_loss_min = validation_loss_avg
                best_epoch = epoch
 
            # Otherwise increment count of epochs with no improvement
            else: 
                epochs_no_improve += 1 
                # Trigger EARLY STOPPING
                if epochs_no_improve >= max_epochs_stop:
                    print(f'\nEarly Stopping! Total epochs (starting from 0): {epoch}. Best epoch: {best_epoch} with loss: {validation_loss_min:.2f}')
                    # Load the best state dict (at the early stopping point)
                    net.load_state_dict(torch.load(output_dir + "/PXS_score_model.pth"))
                    # attach the optimizer
                    net.optimizer = optimizer

                    # save history with pickle
                    with open(output_dir + "/history_training.pckl", "wb") as f:
                        pickle.dump(history, f)

                    f = open(output_dir + "/history_training.txt", 'a+')
                    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" +
                            "Early stopping! Total epochs (starting from 0): {:.0f}\n".format(epoch) +
                            "Best epoch: {:.0f}\n".format(best_epoch) +
                            "Validation loss at best epoch: {:.3f}\n".format(validation_loss_min)
                            )
                    f.close()

                    return net, history, output_dir # break the function

        # after each Epoch

        # append to lists for graphing
        training_losses.append(training_loss_avg)
        validation_losses.append(validation_loss_avg)

        # training euclidean distance stats

        # extract euclidean distances if label is 0 or 1
        euclid_if_0 = [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 0]
        euclid_if_1 = [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 1]
        euclid_if_0 = np.array(euclid_if_0).tolist()
        euclid_if_1 = np.array(euclid_if_1).tolist()
        
        # summary statistics for euclidean distances
        mean_euclid_0t = statistics.mean(euclid_if_0) 
        std_euclid_0t = statistics.pstdev(euclid_if_0) # population stdev
        mean_euclid_1t = statistics.mean(euclid_if_1)
        std_euclid_1t = statistics.pstdev(euclid_if_1) # population stdev
        euclid_diff_t = mean_euclid_1t - mean_euclid_0t

        # validation euclidean distance stats

        # extract euclidean distances if label is 0 or 1
        euclid_if_0 = [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 0]
        euclid_if_1 = [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 1]
        euclid_if_0 = np.array(euclid_if_0).tolist()
        euclid_if_1 = np.array(euclid_if_1).tolist()
        
        # summary statistics for euclidean distances
        mean_euclid_0v = statistics.mean(euclid_if_0) 
        std_euclid_0v = statistics.pstdev(euclid_if_0) # population stdev
        mean_euclid_1v = statistics.mean(euclid_if_1)
        std_euclid_1v = statistics.pstdev(euclid_if_1) # population stdev
        euclid_diff_v = mean_euclid_1v - mean_euclid_0v

        # store in history list
        history = [training_losses, validation_losses, euclid_diff_t, euclid_diff_v]
 
        # save history with pickle
        with open(output_dir + "/history_training.pckl", "wb") as f:
            pickle.dump(history, f)
 
        print("Epoch number: {:.0f}\n".format(epoch),
            "Training loss: {:.3f}\n".format(training_loss_avg),
            "Validation loss: {:.3f}\n".format(validation_loss_avg),
            "\nTraining \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0t),
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0t),
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1t),
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1t),
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_t),
            "\nValidation \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0v),
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0v),
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1v),
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1v),
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_v)
            )

        # write history to file
        f = open(output_dir + "/history_training.txt", 'a+')
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" +
            "Epoch number: {:.0f}\n".format(epoch) +
            "Training loss: {:.3f}\n".format(training_loss_avg) +
            "Validation loss: {:.3f}\n".format(validation_loss_avg) +
            "\nTraining \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0t) +
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0t) +
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1t) +
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1t) +
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_t) +
            "\nValidation \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0v) + 
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0v) + 
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1v) + 
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1v) +
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_v) + "\n"
            )
        f.close()
 
    # Load the best state dict (at the early stopping point)
    net.load_state_dict(torch.load(output_dir + "/PXS_score_model.pth"))
    # After training through all epochs attach the optimizer
    net.optimizer = optimizer

    # Return the best model and history
    print(f'\nAll Epochs completed! Total epochs (starting from 0): {epoch}. Best epoch: {best_epoch} with validation loss: {validation_loss_min:.2f}')

    return net, history, output_dir

# siamese training 
net, history, output_dir = siamese_training(training_dataloader = training_dataloader, 
                                            validation_dataloader = validation_dataloader, 
                                            working_path = working_path,
                                            output_folder_name = 'CheXpert_model')
 
# Training/validation learning curves
plt.title("Number of Training Epochs vs. Contrastive Loss")
plt.xlabel("Training Epochs")
plt.ylabel("Contrastive Loss")
plt.plot(range(0, len(history[0])), history[0], label = "Training loss")
plt.plot(range(0, len(history[1])), history[1], label = "Validation loss")
plt.legend(frameon=False)
plt.savefig(output_dir + "/Learning_curve.png")
plt.close()

  

'''
Testing scripts
'''

img0, img1, label, meta = next(iter(training_dataloader))
concatenated = torch.cat((img0, img1),0)
cat = torchvision.utils.make_grid(concatenated, nrow = 16) # make the nrow the batch size
torchvision.utils.save_image(cat, 'testcat.jpg')



