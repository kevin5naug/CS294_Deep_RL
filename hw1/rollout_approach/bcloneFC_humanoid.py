
# coding: utf-8

# In[5]:

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

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
    def __init__(self, INPUTDIM, OUTPUTDIM):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(INPUTDIM, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc5=nn.Linear(512,OUTPUTDIM)
    
    def forward(self,x):
        x=self.bn1(self.fc1(x))
        x = F.relu(x)
        x=self.bn2(self.fc2(x))
        x = F.relu(x)
        x=self.bn3(self.fc3(x))
        x = F.relu(x)
        x=self.bn4(self.fc4(x))
        x = F.relu(x)
        x=self.fc5(x)
        return x



def init_weights1(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0,0.002)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.002)
        m.bias.data.fill_(0)



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
    parser.add_argument('expert_policy_data', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    #need number of epoch
    args = parser.parse_args()
    
    print('loading expert policy data for training')
    with open(args.expert_policy_data, 'rb') as handle:
        expert_data = pickle.load(handle)
    
    #train the network
    torch.manual_seed(25)
    o_expert=expert_data['observations']
    (N,N_step,N_obs)=o_expert.shape
    a_expert=expert_data['actions']
    (N,N_step,_,N_action)=a_expert.shape
    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit
    net=CNN(N_obs, N_action)
    
    #todo:initilize network parameters
    net.apply(init_weights)

    import torch.optim as optim
    optimizer=optim.Adam(net.parameters(),lr=1e-5, weight_decay=5e-9)
    criterion=nn.MSELoss()
    loss_history=[]
    reward_mean_history=[]
    reward_std_history=[]
    for j in range(args.num_epochs):
        print("epoch %i"%j)
        net.train()
        (N,N_step,N_obs)=o_expert.shape
        (N,N_step,_,N_action)=a_expert.shape
        for k in range(max_steps):
            optimizer.zero_grad()
            index=k
            o=Variable(torch.from_numpy(o_expert[:,index,:]).reshape(N,N_obs))
            o=o.float()
            a_out=net.forward(o)
            a_label=torch.from_numpy(a_expert[:,index,:].reshape(N,N_action))
            loss=criterion(a_out.float(), a_label.float())
            loss.backward()
            optimizer.step()
        print("No DAGGER")
        print(loss/N)
        loss_history.append(loss/N)
        
        #test the network
        with tf.Session():
            tf_util.initialize()

            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit
            net.eval()

            r_new=[]
            for i in range (int(args.num_rollouts)//4):
                totalr=0
                obs=env.reset()
                done=False
                steps=0
                while not done:
                    obs=Variable(torch.Tensor(obs).reshape(1,N_obs))
                    action_new=net.forward(obs).detach().numpy()
                    obs,r,done,_=env.step(action_new.reshape(N_action))
                    totalr+=r
                    steps+=1
                    if steps >= max_steps:
                        break
                r_new.append(totalr)
            u=np.average(np.array(r_new))
            sigma=np.std(np.array(r_new))
            reward_mean_history.append(u)
            reward_std_history.append(sigma)
            print('current reward mean', u)
            print('current reward std', sigma)
    fig0=plt.figure(0)
    plt.plot(loss_history, '-o')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    fig0.savefig('/Users/joker/imitation_learning/humanoid.png')
    
    reward_mean_history=np.array(reward_mean_history)
    reward_std_history=np.array(reward_std_history)
    #print(reward_mean_history.shape)
    #print(reward_std_history.shape)
    print('mean:', reward_mean_history)
    print('std:', reward_std_history)
    
    fig1=plt.figure(1)
    plt.errorbar(np.arange(args.num_epochs),reward_mean_history, reward_std_history, marker="s", mfc='blue', mec='yellow')
    fig1.savefig('/Users/joker/imitation_learning/humanoid_reward.png')
    
main()
