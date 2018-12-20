#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import torch
from PIL import Image
from data_load import ImageNet2012Dataset, RandomCrop, Rescale, ToTensor
from torchvision import transforms
import cv2

import json
with open('./labels/indices.json') as f:
    labels = json.load(f)
    
import os
import sys
module_path = os.path.abspath(os.path.join('./models'))
if module_path not in sys.path:
    sys.path.append(module_path)

def center_crop(image, new_h, new_w):
    h, w = image.shape[:2]
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    image = image[top:top + new_h, left:left + new_w]
    return image

def to_tensor(image):
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    image = image.float()
    return image

def predict(net, img):
    with torch.no_grad():
        output = net(img)
        max_vals, max_indices = torch.topk(output, 5)
        max_indices = torch.squeeze(max_indices, 0)
        max_indices = max_indices.tolist()
        bests = [labels.get(str(i)) for i in max_indices]
        for best in bests:
            print(best)


# In[5]:


checkpoint = torch.load(
    './saved_models/alexnet2-2018-12-20T04_27_47-epoch-33.pt', map_location='cpu')

from alexnet_v2 import AlexNetV2
net = AlexNetV2()
net.load_state_dict(checkpoint['model'])
net.eval()

losses = [i/10 for i in checkpoint["loss_logger"]]
batches = [i for i in range(1, len(losses) + 1)]

plt.figure(figsize=(15,5))
plt.title('alexnet V2')
plt.plot(batches, losses)

def preprocess_alexnet2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = center_crop(img, 224, 224)
    plt.imshow(img)
    img = to_tensor(img)
    img = torch.unsqueeze(img, 0)
    return img


# In[6]:


img1 = cv2.imread('./test_images/n01692333_1000.jpeg')
img1 = preprocess_alexnet2(img1)
predict(net, img1)


# In[7]:


img2 = cv2.imread('./test_images/n01440764_10026.jpeg')
img2 = preprocess_alexnet2(img2)
predict(net, img2)


# In[8]:


checkpoint = torch.load(
    './saved_models/vgg16-2018-12-19T06_12_17-epoch-29.pt', map_location='cpu')

from vgg16 import VGG16
net = VGG16()
net.load_state_dict(checkpoint['model'])
net.eval()


losses = [i/10 for i in checkpoint["loss_logger"]]
batches = [i for i in range(1, len(losses) + 1)]

plt.figure(figsize=(15,5))
plt.title('vgg16')
plt.plot(batches, losses)

def preprocess_vgg16(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = center_crop(img, 224, 224)
    plt.imshow(img)
    img = to_tensor(img)
    img = torch.unsqueeze(img, 0)
    return img


# In[10]:


img = cv2.imread('./test_images/cat2.jpg')
img = preprocess_vgg16(img)
predict(net, img)


# In[ ]:




