from models.lr_gan_model import LRGANModel
import toml
from data import LipreadingDataset_LR as LipreadingDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import os
import pdb
import math
import sys
import torch
import time
import numpy as np

with open(sys.argv[1], 'r') as optFile:
    opt = toml.loads(optFile.read())

val_dataset = LipreadingDataset(opt["validation"]["dataset"],  "val")
val_dataloader = DataLoader(
                                    val_dataset,
                                    batch_size=opt["input"]["batchsize"],
                                    shuffle=opt["input"]["shuffle"],
                                    num_workers=opt["input"]["numworkers"],
                                    drop_last=False
                                )

model = LRGANModel()
model.initialize(opt)
print('Start Testing...')

model.set_eval()
count = np.array([0, 0, 0, 0])
acc = np.array([0.0, 0.0, 0.0, 0.0])
val_loss = np.array([0.0, 0.0, 0.0, 0.0])
len_dataset = len(val_dataloader)
with torch.no_grad():
    for j , data in enumerate(val_dataloader):
        print(j,'/', len_dataset)
        model.set_val_input(data)
        cnt, loss = model.validate_fusion()
        count += cnt 
        val_loss += loss
    acc = count/len(val_dataset)
    val_loss = val_loss/ len(val_dataloader)
    for k in range(acc.shape[0]):
        print('Acc_{} is : '.format(k), count[k], acc[k])
        print('Val_loss_{} is : '.format(k),val_loss[k])