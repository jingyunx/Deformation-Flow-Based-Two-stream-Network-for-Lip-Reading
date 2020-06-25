import os
import numpy as np
import glob
import time
import cv2
from torch.utils.data import Dataset
from .cvtransforms import *
import torch
import pickle
import getpass
import random 
random.seed()

def load_file(filename):
    arrays = np.load(filename)
    arrays = arrays / 255.
    return arrays

class LRWDataset(Dataset):
    def __init__(self, path, set):
        self.set = set 
        self.id_pkl = 'data/all_train.pkl'
        if set == 'val':
            self.id_pkl = 'data/all_test.pkl'
        self.file_list = pickle.load(open(self.id_pkl, 'rb'))
        print('Total num of samples: ', len(self.file_list))

    def __getitem__(self, idx):
        path = self.file_list[idx][1]  
        inputs = load_file(path)
        if(self.set == 'train'):
            batch_img = RandomCrop(inputs, (88, 88))
        batch_img = inputs
        label = self.file_list[idx][0]
        vid_tensor =  torch.FloatTensor(batch_img[:, np.newaxis,...])
        sample =  {'x': vid_tensor, 'label': torch.LongTensor([int(label)])}
        return sample 

    def __len__(self):
        return len(self.file_list)