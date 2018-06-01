
# coding: utf-8

# In[5]:

import pickle
import tensorflow as tf
import numpy as np
import gym

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as ini
import random


# In[6]:

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1=nn.Conv1d(1,512,376)
        self.conv2=nn.Conv1d(512,512,1)
        self.conv3=nn.Conv1d(512,256,1)
        self.conv4=nn.Conv1d(256,128,1)
        self.conv5=nn.Conv1d(128,17,1)
    
    def forward(self,x):
        x=self.conv1(x)
        x = F.relu(x)
        x=self.conv2(x)
        x = F.relu(x)
        x=self.conv3(x)
        x = F.relu(x)
        x=self.conv4(x)
        x = F.relu(x)
        x=self.conv5(x)
        x=x.view(-1,17)
        return x


# In[3]:

# test structure
# net=CNN()
# x=Variable(torch.randn(1,1,376),requires_grad=True)
# y_pred=net.forward(x)
# print(y_pred)


# In[4]:

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    #need number of epoch
    args = parser.parse_args()
    
    print('loading expert policy data for training')
    with open(args.expert_policy_file, 'rb') as handle:
        expert_data = pickle.load(handle)
    o_expert=expert_data['observations']
    a_expert=expert_data['actions']
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    rollout_list=list(range(args.num_rollouts))
    
    net=CNN()
    
    #todo:initilize network parameters
    
    import torch.optim as optim
    optimizer=optim.Adam(net.parameters(),lr=5e-4, weight_decay=5e-7)
    criterion=nn.CrossEntropyLoss()
    loss_history=[]
    for j in range(args.num_epochs):
        random.shuffle(rollout_list)
        for i in rollout_list:
            print("epoch %i iteration %i"%(j,i))
            for k in range(max_steps):
                index=i*max_steps+k
                o=Variable(torch.from_numpy(o_expert[index]).reshape(1,1,376))
                o=o.float()
                a_out=net.forward(o)
                a_label=torch.from_numpy(a_expert[index])
                a_label=a_label.long()
                loss=criterion(a_out, torch.max(a_label,1)[1])
                loss.backward()
                loss_history.append(loss)
                optimizer.step()
            print(loss)
    
    plt.plot(loss_history, '-o')
    plt.xlabel('iteration')
    plt.ylabel('loss')

main()
