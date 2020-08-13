#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torch.autograd import Variable


# In[ ]:


test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


model_ft = torch.load('./models/dit_08_06.h5')
model_ft = model_ft.to(device)
model_ft.eval()
f = open('./models/dit_08_06.txt','r')
class_name=f.readline().split(',')
f.close()


# In[ ]:


def predict_image(image):
    with torch.no_grad():
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = model_ft(input)
        index = output.data.cpu().numpy().argmax()
        return class_name[index]


# In[ ]:


testset = 'testset'
images = os.listdir(testset)
for image in images:
    img = Image.open(os.path.join(testset, image))
    result = predict_image(img)
    print(result)
    plt.imshow(img)
    plt.show()
    print('='*50)


# In[ ]:




