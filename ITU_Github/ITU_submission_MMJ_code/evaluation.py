#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np
import pandas as pd
import scipy.io as scipyio
import matplotlib.pyplot as plt
from scipy import stats
from plyfile import PlyData, PlyElement
from tqdm import tqdm

import math
from math import *

import pylab
from scipy.signal import find_peaks

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import argparse
import logging
import csv
from models.Simple_NN import *
#import models


# Read HDF5 file.
def read_hdf5(path,x) :
    f = h5py.File(path, "r")   
    # Print the keys of groups and datasets under '/'.
    dataset_name = [key for key in f.keys()]
    #print(dataset_name, "\n") 
    
    # xxxx_0 represents data from scenario 32
    # xxxx_1 represents data from scenario 33
    # xxxx_2 represents data from scenario 34
    # xxxx_3 represents data from scenario 31
    if x == '32' :
        d1 = f["beam_index_0"]
        d2 = f["camera_bb_0"]
        d3 = f[f"gt_0"] # gt
    elif x == '33' :
        d1 = f["beam_index_1"]
        d2 = f["camera_bb_1"]
        d3 = f[f"gt_1"] # gt
    elif x == '34' :
        d1 = f["beam_index_2"]
        d2 = f["camera_bb_2"]
        d3 = f[f"gt_2"] # gt
    elif x == '31' :
        d1 = f["beam_index_3"]
        d2 = f["camera_bb_3"]
        d3 = f[f"pos_3"] # gt
    
    gt_pos = d3[:]
    camera_bb = d2[:]
    beam_index = d1[:]
    
    n_samples = len(beam_index)
    
    input_data = np.zeros((n_samples, 24))
    output_data = np.zeros((n_samples, 64))
    
    for i in range(n_samples) :
        a = gt_pos[i].reshape(-1,)
        b = camera_bb[i].reshape(-1,)
        input_data[i] = np.hstack((a, b))
        
        arr = np.zeros((64,1))
        arr[int(beam_index[i])] = 1
        arr = arr.reshape(-1,)
        arr = arr.astype(np.float32)

        output_data[i] = arr
        
    return input_data, output_data

class beampower_pos_loader(Dataset):
    def __init__(self, Data, Target):
        super(beampower_pos_loader, self).__init__()
        self.Data = Data
        self.Target = Target

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        target = self.Target[index]
        return data.astype(np.float32), target.astype(np.float32)
    
def create_data(batch_size, x) :

    test_data_path = "./ITU_test_dataset.hdf5"
    input_data_test, output_data_test = read_hdf5(test_data_path, x)
        
    # Create Dataset
    test_dataset = beampower_pos_loader(input_data_test, output_data_test)
    
    # Create Loader
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False, num_workers=4)
 
    return test_loader

def main(args) :
    #print('train model ...')
    Epochs = 200
    device = 0
    batch_size_arr = [32] #, 64
    
    # x can be set to the corresponding scenario number 31, 32, 33, 34
    x_arr = ['31', '32', '33', '34']
    for x in x_arr :

        net = torch.load(f'models/model_{x}.pt')  
        #net = model_object.load_state_dict(torch.load(f'model_param_{x}.pkl'))
        # net = Simple_NN()
        # model_dict = torch.load(f'model_param_{x}.pt')
        # net.load_state_dict(model_dict, strict=False)
        net = net.to(device)

        for batch_size in batch_size_arr :

            test_loader = create_data(batch_size, x)
            test_output_arr = []

            net.eval()

            for batch_index, (data, target) in enumerate(test_loader):

                # Check if its not matching
                if data.shape[0] != 1:
                    continue
                # Shift to gpu
                data = data.cuda(device)
                target = target.cuda(device)
                # Forward Pass
                output = net(data)

                output = output.cpu().data.numpy()
                test_output_arr.append(output)

        test_output_arr = np.asarray(test_output_arr).reshape(-1,64)
        #print(test_output_arr.shape)
        np.save(file=f"NN_with_0_output_arr_test_{x}.npy", arr=test_output_arr)
 
    f = h5py.File("./ITU_test_dataset.hdf5", "r")   
    d = f["dataset_id"]
    dataset_id = d[:]
    dataset_id = dataset_id.reshape(-1,)
    n_samples = dataset_id.shape[0]

    test_result_arr = np.zeros((n_samples, 64))
    
    test_result_arr[dataset_id==0] = np.load(file="./NN_with_0_output_arr_test_32.npy")
    test_result_arr[dataset_id==1] = np.load(file="./NN_with_0_output_arr_test_33.npy")
    test_result_arr[dataset_id==2] = np.load(file="./NN_with_0_output_arr_test_34.npy")
    test_result_arr[dataset_id==3] = np.load(file="./NN_with_0_output_arr_test_31.npy")

    np.save(file=f"NN_with_0_output_arr_test_arr.npy", arr=test_result_arr)

    output_arr = np.load(file=f"NN_with_0_output_arr_test_arr.npy") 
    #print(output_arr.shape)

    outputs = torch.from_numpy(output_arr)
    _, outputs_top1 = outputs.topk(1, dim=1, largest=True)
    outputs_top1 = outputs_top1.numpy()

    _, outputs_top3 = outputs.topk(3, dim=1, largest=True)
    outputs_top3 = outputs_top3.numpy()

    n_samples = len(outputs_top3)

    samples = np.linspace(0, n_samples-1, n_samples)
    samples = samples +1
    samples = samples.reshape(-1,1)
    samples = samples.astype(int)
    outputs_top3 = outputs_top3.astype(int)
    ones = np.asarray([[1]*3 for _ in range(len(outputs_top3))])
    outputs_top3 = outputs_top3 + ones  # +1
    arr = np.hstack((samples, outputs_top3))
    
    with open('test.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "top-1 beam", "top-2 beam", "top-3 beam"])
        writer.writerows(arr)
    
    print('finish')
 # calculate accuracy for top1 and top3
def cal_accuracy(output, target) :
    output_arr = np.asarray(output).reshape(-1,64)
    target_arr = np.asarray(target).reshape(-1,64)

    outputs = torch.from_numpy(output_arr)
    targets = torch.from_numpy(target_arr)
    _, outputs_top1 = outputs.topk(1, dim=1, largest=True)
    _, targets = targets.topk(1, dim=1, largest=True)
    outputs_top1 = outputs_top1.numpy()
    targets = targets.numpy()
    
    _, outputs_top3 = outputs.topk(3, dim=1, largest=True)
    outputs_top3 = outputs_top3.numpy()

    test_accuracy_top1 = 100. * sum(outputs_top1[i] == targets[i] for i in range(len(targets)))/len(targets)
    test_accuracy_top3 = 100. * sum(targets[i] in outputs_top3[i] for i in range(len(targets)))/len(targets)

    return test_accuracy_top1[0], test_accuracy_top3

# calculate DBA-Score for top 3
def top_3(output, target) :
    output_arr = np.asarray(output).reshape(-1,64)
    target_arr = np.asarray(target).reshape(-1,64)
    
    outputs = torch.from_numpy(output_arr)
    targets = torch.from_numpy(target_arr)
    _, outputs_top1 = outputs.topk(1, dim=1, largest=True)
    _, targets = targets.topk(1, dim=1, largest=True)
    outputs_top1 = outputs_top1.numpy()
    targets = targets.numpy()
    
    _, outputs_top2 = outputs.topk(2, dim=1, largest=True)
    outputs_top2 = outputs_top2.numpy()
    
    _, outputs_top3 = outputs.topk(3, dim=1, largest=True)
    outputs_top3 = outputs_top3.numpy()
    
    target_index = targets.reshape(-1,1)
    outputs_top3 = outputs_top3.reshape(-1,3)
    score_top3 = DBA_score(target_index, outputs_top3)
    
    return score_top3

# calculate DBA-Score
def DBA_score(ground_truth, predictions) :
    N = predictions.shape[0]
    K = predictions.shape[1]
    delta = 5
    Y = [0]*K
    for k in range(K) :
        for i in range(N) :
            if k >= 2 :
                Y[k] += min(min(abs(predictions[i][k] - ground_truth[i])/delta, 1), min(abs(predictions[i][k-1] - ground_truth[i])/delta, 1), min(abs(predictions[i][k-2] - ground_truth[i])/delta, 1))
            elif k >= 1 :
                Y[k] += min(min(abs(predictions[i][k] - ground_truth[i])/delta, 1), min(abs(predictions[i][k-1] - ground_truth[i])/delta, 1))
            else :
                Y[k] += min(abs(predictions[i][k] - ground_truth[i])/delta, 1)

        Y[k] = 1 - Y[k]/N
    score = sum(Y)/K
    return score[0]


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ITU NN')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--batch_num", type=int, default=None)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    try:
        params = get_params()
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
