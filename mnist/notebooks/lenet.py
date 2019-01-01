#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from data_load import MnistDataset
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt


# In[2]:


train_dataset = MnistDataset(
    images_path='../dataset/train-images-idx3-ubyte',
    labels_path='../dataset/train-labels-idx1-ubyte',
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0)


# In[5]:


for batch_i, (images, labels) in enumerate(train_loader):
    if batch_i == 1:
        print(images, labels)
        print(labels[0])
        plt.imshow(images[0])
        


# In[ ]:




