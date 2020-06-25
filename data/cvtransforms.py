# coding: utf-8
import random
import cv2
import numpy as np


def CenterCrop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    img = np.zeros((batch_img.shape[0], th, tw))
    x1 = int(round((w - tw))/2.)
    y1 = int(round((h - th))/2.)    
    img = batch_img[:, y1:y1+th, x1:x1+tw]
    return img

def RandomCrop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    img = np.zeros((batch_img.shape[0], th, tw))
    x1 = random.randint(0, 8)
    y1 = random.randint(0, 8)
    img = batch_img[:,y1:y1+th,x1:x1+tw]
    return img

def HorizontalFlip(batch_img):
    if random.random() > 0.5:
        batch_img = cv2.flip(batch_img, 2)
    return batch_img

def RandomDrop(batch_img):
    i = 0
    for j in range(batch_img.shape[0]):
        if random.random() > 0.01:
            batch_img[j] = batch_img[i]
            i += 1
    i = min(i, batch_img.shape[0] - 1)
    for j in range(i, batch_img.shape[0]):
        batch_img[j] = batch_img[i]
    return batch_img
    
def ColorNormalize(batch_img):
    mean = 0.413621
    std = 0.1700239
    batch_img = (batch_img - mean) / std
    return batch_img
