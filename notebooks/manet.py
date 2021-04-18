#!/usr/bin/env python
# coding: utf-8

# # Pix2Pix

# **기억할 점**
# - **한 도메인에 있는 이미지를 다른 도메인으로 해석해보자**는 관점
# - Image to Image Mapping Network에서 Photo-realistic을 추구하고자 함
# - GAN의 adverserial을 사용
# - U-Net과 PatchGAN 등을 이용해서 성능을 최적화함
# - 양쪽 도메인에 대응되는 데이터 쌍이 존재해야 함(반면 cyclegan은 데이터 쌍이 없어도 됨)

# ## Pix2Pix의 손실함수
# - Adverserial Loss
# 
# $$L_{cGAN}(G, D) = E_{y\sim p_{data}(y)} \big[\text{log}D(y) \big] + E_{x\sim P_{data}(x), z \sim P_{z}(z)} \big[\| y - G(x, z) \|_{1}\big]$$
# 
# 
# - Reconstruction Loss
# 
# $$L_{L1}\big(G \big) = E_{x, y \sim P_{data}(x, y), z \sim p_{z}(z)} \big[ \| y - G(x, z) \|_{1} \big]$$
# 
# - Total Loss
# 
# $$G^{*} = arg \underset{G}{\operatorname{min}} \underset{D}{\operatorname{max}} L_{cGAN}(G, D) + \lambda L_{L1}(G)$$

# ## 1. Import Libs

# In[2]:


from __future__ import print_function
import argparse
import os
from math import log10

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from networks import define_G, define_D, GANLoss, print_network
from data import get_training_set, get_test_set
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter


# ## 2. Setting Hyperparameters

# In[3]:


parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=False, default='facades',help='facades')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
params = parser.parse_args([])

print(params)

if params.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(params.seed)
if params.cuda:
    torch.cuda.manual_seed(params.seed)


# In[ ]:





# ## 3. Load Dataset

# In[4]:


root_path = '/home/nisnab/workspace/ImageTranslation/Pix2Pix/dataset/'
train_set = get_training_set(root_path + params.dataset)
test_set = get_test_set(root_path + params.dataset)

train_data_loader = DataLoader(dataset=train_set, num_workers=params.threads, batch_size=params.batchSize, shuffle=True)
test_data_loader = DataLoader(dataset=test_set, num_workers=params.threads, batch_size=params.batchSize, shuffle=False)


# In[5]:


test_data_loader


# ## 4. Build Model & Optimizers & Criterions
# ### 4.1 Build Model

# In[6]:


netG = define_G(params.input_nc, params.output_nc, params.ngf, 'batch', False, [0])
netD = define_D(params.input_nc + params.output_nc, params.ndf, 'batch', False, [0])

#print(netG)
#print(netD)


# ### 4.2 Optimizers

# In[7]:


optimizerG = optim.Adam(netG.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
writer=SummaryWriter(f'logs/tensorlogs')


# ### 4.3 Loss Functions

# In[8]:


criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

real_a = torch.FloatTensor(params.batchSize, params.input_nc, 256, 256)
real_b = torch.FloatTensor(params.batchSize, params.output_nc, 256, 256)

if params.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)


# ## 5. Training Models

# In[9]:


G_losses=[]
D_losses=[]

for epoch in range(1, params.nEpochs + 1):
    for iteration, batch in enumerate(train_data_loader, 1):
        # forward
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        
        fake_b = netG(real_a)
        
        ## Discriminator network : max log(D(x, y)) + log(1 - D(x, G(x)))
        optimizerD.zero_grad()
        
        # train fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)
        
        # train real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # combine loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        
        loss_d.backward()
        
        optimizerD.step()
        
        ## Generator network : max log(D(x, G(x))) + L1(y, G(x))
        optimizerG.zero_grad()
        
        # 1. G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)
        
        # 2. G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * params.lamb
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        optimizerG.step()
        
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(train_data_loader), loss_d.data, loss_g.data))
        
        G_losses.append(loss_g.data)
        D_losses.append(loss_d.data)
        
        """
    
    avg_psnr = 0
    for batch in test_data_loader:
        input2, target = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True)
        if params.cuda:
            input2 = input2.cuda()
            target = target.cuda()

        prediction = netG(input2)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
        #print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data_loader)))

    if epoch % 3 == 0:
        if not os.path.exists('save_model'):
            os.mkdir('save_model')
        if not os.path.exists(os.path.join('save_model', params.dataset)):
            os.mkdir(os.path.join('save_model', params.dataset))
        net_g_model_out_path = "save_model/{}/netG_model_epoch_{}.pkl".format(params.dataset, epoch)
        net_d_model_out_path = 'save_model/{}/netD_model_epoch_{}.pkl'.format(params.dataset, epoch)
        torch.save(netG, net_g_model_out_path)
        torch.save(netD, net_d_model_out_path)
        print("model saved to {}".format("model " + params.dataset))"""


# In[ ]:


#loss_g


# In[ ]:





# In[10]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[11]:


"""if __name='__main__':
    print(args.dataset)"""


# In[ ]:





# In[ ]:





# In[ ]:




