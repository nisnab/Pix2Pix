#!/usr/bin/env python
# coding: utf-8

# ## 1. Import Libs

# In[17]:


from __future__ import print_function
import argparse
import os

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from util import is_image_file, load_img, save_img


# ## 2. Setting Hyperparameters

# In[18]:


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades', help='facades')
parser.add_argument('--model', required=False, type=str, default='save_model/facades/netG_model_epoch_500.pkl', help='model file to use')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
params = parser.parse_args([])
print(params)


# ## 3. Testing

# In[19]:


netG = torch.load(params.model)


# In[20]:


image_dir = 'dataset/{}/test/a/'.format(params.dataset)
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

# Preprocessing
transform_list = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, .5), (.5, .5, .5))]
transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    
    # 모델에 맞는 텐서구조로 변경
    input = Variable(img, volatile=True).view(1, -1, 256, 256)
    
    if params.cuda:
        netG = netG.cuda()
        input = input.cuda()
    
    out = netG(input)
    out = out.cpu()
    out_img = out.data[0]
    
    # generate한 이미지를 저장하는 파일을 생성
    if not os.path.exists(os.path.join("result", params.dataset)):
        os.mkdir(os.path.join('result', params.dataset))
    save_img(out_img, "result/{}/{}".format(params.dataset, image_name))


