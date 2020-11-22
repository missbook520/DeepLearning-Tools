import torch
import matplotlib.pyplot as plt
#%matplotlib inline
from torch.optim import *
import torch.nn as nn
import numpy as np


class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,10)
    def forward(self,x):
        return self.fc(x)

#################################
# 手动调整
# model = net()
# LR = 0.01
# optimizer = Adam(model.parameters(),lr = LR)
# lr_list = []
# for epoch in range(100):
#     if epoch % 5 == 0:
#         for p in optimizer.param_groups:
#             p['lr'] *= 0.9
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
# plt.plot(range(100),lr_list,color = 'r')

##################################
# LambdaLR
# lr_list = []
# model = net()
# LR = 0.01
# optimizer = Adam(model.parameters(),lr = LR)
# lambda1 = lambda epoch:np.sin(epoch) / epoch
# scheduler = lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)
# for epoch in range(100):
#     scheduler.step()
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
# plt.plot(range(100),lr_list,color = 'r')

##################################
# StepLR
# lr_list = []
# model = net()
# LR = 0.01
# optimizer = Adam(model.parameters(),lr = LR)
# scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.8)
# for epoch in range(100):
#     scheduler.step()
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
# plt.plot(range(100),lr_list,color = 'r')

##################################
# MultiStepLR
# lr_list = []
# model = net()
# LR = 0.01
# optimizer = Adam(model.parameters(),lr = LR)
# scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[20,80],gamma = 0.9)
# for epoch in range(100):
#     scheduler.step()
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
# plt.plot(range(100),lr_list,color = 'r')

####################################
# ExponentialLR
# lr_list = []
# model = net()
# LR = 0.01
# optimizer = Adam(model.parameters(),lr = LR)
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# for epoch in range(100):
#     scheduler.step()
#     lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
# plt.plot(range(100),lr_list,color = 'r')

####################################
# CosineAnnealingLR
lr_list = []
model = net()
LR = 0.1
Warm_epoch = 20
optimizer = Adam(model.parameters(),lr = LR)
#scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[60,80],gamma = 0.1)
for epoch in range(100):
    if epoch <= Warm_epoch:
        lr = 1e-6 + (LR - 1e-6) * (epoch / Warm_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch > Warm_epoch:
        scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')

####################################

plt.show()