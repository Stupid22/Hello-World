#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:57:35 2019

@author: Starry-9t-Stupid22
"""

import os
import numpy as np
import pickle
import itertools

import torch
import torch.nn as nn
#import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
###################################################
#-------------MODEL-------------------------------#
###################################################

torch.manual_seed(2019)

class VAE(nn.Module):
    def __init__(self, dimZ, dimX):
        super(VAE, self).__init__()
        self.dimZ = dimZ
        self.fc_mean = nn.Linear(dimX, dimZ)
        self.fc_sigma = nn.Linear(dimX, dimZ)
        #self.fc_d1 = nn.Linear(dimZ, dimZ)
        self.fc_g = nn.Linear(dimZ, dimX)
        
    def encoder(self, x):
        mean = self.fc_mean(x)
        sigma = self.fc_sigma(x)
        return mean, sigma
        
    def decoder(self, z):
        return torch.sigmoid(self.fc_g(z))
    
    def sample_normal(self, mean, sigma):
        epsilon = Variable(torch.randn(self.dimZ))
        z = epsilon.mul(sigma).add_(mean)
        return z
    
    def parameters(self):
        return itertools.chain(*[self.fc_mean.parameters(),
                                 self.fc_sigma.parameters(),
                                 self.fc_g.parameters()])
    
    def __call__(self, x):
        mean, sigma = self.encoder(x)
        z = self.sample_normal(mean, sigma)
        y = self.decoder(z)
        return y, mean, sigma
        
def getLoss(x, y, miu, sigma):
    res = torch.abs(y - x)
    reLoss = torch.sum(torch.pow(res, 2)) / (x.size(0))    
    KLLoss = 0.5 * torch.mean(torch.sum(1 + sigma - (miu ** 2) - torch.exp(sigma)))
    return reLoss, KLLoss, reLoss - KLLoss
        
def train(model, optimizer, dataloader, numEpoch):
    losses = []
    res = []
    kls = []
    i = 0
    m1 = []
    m2 = []
    m3 = []
    for epoch in range(numEpoch):
        i += 1
#        if i==4:
#            break
        s1 = torch.sum(model.fc_mean.weight)
        #s2 = torch.sum(model.fc_sigma.weight)
        s2 = model.fc_sigma.weight.data
        s3 = torch.sum(model.fc_g.weight)
        for batchIdx, (data,target) in enumerate(dataloader):
            optimizer.zero_grad()
            y, miu, sigma = model(data)
            re, kl, loss = getLoss(data, y, miu, sigma)
            loss.backward()
            optimizer.step()
        ss1 = torch.sum(model.fc_mean.weight)
        #ss2 = torch.sum(model.fc_sigma.weight)
        ss2 = model.fc_sigma.weight.data
        ss3 = torch.sum(model.fc_g.weight)
        res.append(re)
        kls.append(kl)
        losses.append(float(loss.data))
        m1.append(s1-ss1)
        m2.append(torch.sum(torch.abs(s2-ss2)))
        m3.append(s3-ss3)
#        print(model.fc_g.weight)
    return losses, res, kls, m1, m2, m3

X = np.load('X.npy')
print(X.shape)

X = torch.from_numpy(X.astype(np.float32))
X = Variable(X)

trainloader = DataLoader(
        TensorDataset(X[:1280], torch.randn(1280)),
        batch_size = 1280,
        shuffle = True,
        drop_last = True
        )
        
testloader = DataLoader(
        TensorDataset(X[1280:2560], torch.randn(X[1280:2560].size(0))),
        batch_size = 1280,
        shuffle = True,
        drop_last = True
        )

numEpoch = 100
model = VAE(5, 1738)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.01) 
L, res, kls, m1, m2, m3 = train(model, optimizer, trainloader, numEpoch)

#print(L)   
 
##################################################
#                                                # draw graph by tensorboard
#lossWriter = SummaryWriter()   
#for loss, x in zip(L, range(numEpoch)):
#    lossWriter.add_scalar('loss', loss, x)
#
#'''commond line to see graph
#tensorboard --logdir .runs
#'''
###################################################
#try to see the change of weight, couldn't get it# draw graph by pyplot
fig = plt.figure()
p1 = fig.add_subplot(231)
p1.plot(L)
p2 = fig.add_subplot(232)
p2.plot(res)
p3 = fig.add_subplot(233)
p3.plot(kls)
p4 = fig.add_subplot(234)
p4.plot(m1)
p5 = fig.add_subplot(235)
p5.plot(m2)
p6 = fig.add_subplot(236)
p6.plot(m3)
plt.show()

