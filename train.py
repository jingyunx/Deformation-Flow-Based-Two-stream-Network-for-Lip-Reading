from models.lr_gan_model import LRGANModel
from data import LipreadingDataset_LR as LipreadingDataset
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
import toml
import os
import pdb
import math
import sys
import torch
import time
import numpy as np

with open(sys.argv[1], 'r') as optFile:
    opt = toml.loads(optFile.read())

if(opt["general"]["usecudnnbenchmark"] and opt["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

writer = SummaryWriter(comment =opt["general"]["comment"])    

train_dataset = LipreadingDataset(opt["training"]["dataset"], "train")
train_dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=opt["input"]["batchsize"],
                                    shuffle=opt["input"]["shuffle"],
                                    num_workers=opt["input"]["numworkers"],
                                    drop_last=False
                                    )

val_dataset = LipreadingDataset(opt["validation"]["dataset"],  "val")
val_dataloader = DataLoader(
                                val_dataset,
                                batch_size=opt["input"]["batchsize"],
                                shuffle=opt["input"]["shuffle"],
                                num_workers= opt["input"]["numworkers"],
                                drop_last=False
                                )

model = LRGANModel()
model.initialize(opt)
total_iter = len(train_dataloader)
start_epoch = opt["training"]["startepoch"]
current_iter = 0
num_save= 2

for epoch in range(start_epoch, start_epoch+20):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0
    print('epoch', epoch, epoch_start_time)
    model.set_train()
    for i, data in enumerate(train_dataloader):
        print('e{}:{} / {}'.format(epoch, i, total_iter), end = ' ')
        model.set_input(data)
        model.train_fusion() # can be train_DFN()/ train_baseline()/ train_df()
        losses = model.get_current_losses()
        for k,v in losses.items():
            writer.add_scalar('loss_'+k, v, current_iter)
        current_iter += 1 
    print('Start Validation...')    
    model.set_eval()
    count = np.array([0, 0, 0, 0])
    acc = np.array([0.0, 0.0, 0.0, 0.0])
    val_loss = np.array([0.0, 0.0, 0.0, 0.0])
    with torch.no_grad():
        for j , data in enumerate(val_dataloader):
            model.set_input(data)
            cnt, loss = model.validate_fusion() 
            count += cnt 
            val_loss += loss
        acc = count/len(val_dataset)
        val_loss = val_loss/ len(val_dataloader)
        for k in range(acc.shape[0]):
            print('Acc_{} is : '.format(k), count[k], acc[k])
            print('Val_loss_{} is : '.format(k),val_loss[k])
            writer.add_scalar('acc_{}'.format(k), acc[k], current_iter)
            writer.add_scalar('val_loss_{}'.format(k), val_loss[k], current_iter)
        writer.add_scalar('epoch', epoch, current_iter)
    model.save_networks('epoch'+str(epoch)+'_'+str(acc).replace(' ','-')+'_'+str(val_loss).replace(' ','-'))
    print('end of epoch %d'%(epoch))
