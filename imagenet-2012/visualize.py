#!/usr/bin/env python
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import torch

checkpoint = torch.load(
    './saved_models/vgg16-2018-12-09T19_15_19-epoch-13.pt', map_location='cpu')

losses = [i/10 for i in checkpoint["loss_logger"]]
batches = [i for i in range(1, len(losses) + 1)]

plt.figure(figsize=(15,5))
plt.title('vgg16')
plt.plot(batches, losses)


# In[10]:


checkpoint = torch.load(
    './saved_models/alexnet2-2018-12-11T07_58_40-epoch-11.pt', map_location='cpu')

losses = [i/10 for i in checkpoint["loss_logger"]]
batches = [i for i in range(1, len(losses) + 1)]

plt.figure(figsize=(15,5))
plt.title('alexnet2')
plt.plot(batches, losses)


# In[ ]:




