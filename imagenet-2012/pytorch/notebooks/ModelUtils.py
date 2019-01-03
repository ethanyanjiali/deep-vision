#!/usr/bin/env python
# coding: utf-8

# # remove module prefix from DataParallel

# In[1]:


import torch
checkpoint = torch.load(
    '../saved_models/resnet34-yanjiali-010319-epoch-129.pt', map_location='cpu')

state_dict = checkpoint['model']

print(state_dict.keys())

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove 'module.' of dataparallel
    new_state_dict[name]=v
    
print(new_state_dict.keys())

torch.save({
    'epoch':
    checkpoint['epoch'],
    'model':
    new_state_dict,
    'optimizer':
    checkpoint['optimizer'],
    'scheduler':
    checkpoint['scheduler'],
    'loss_logger':
    checkpoint['loss_logger'],
    'acc_logger':
    checkpoint['acc_logger'],
}, '../saved_models/resnet34-yanjiali-010319-epoch-129.pt')


# In[ ]:




