
# coding: utf-8

# In[ ]:


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

class CNN(nn.Module):
    def __init__(self, INPUTDIM, OUTPUTDIM):
        super(CNN, self).__init__()
        self.conv1=nn.Conv1d(1,512,INPUTDIM)
        self.conv2=nn.Conv1d(512,512,1)
        self.conv3=nn.Conv1d(512,256,1)
        self.conv4=nn.Conv1d(256,128,1)
        self.conv5=nn.Conv1d(128,OUTPUTDIM,1)
    
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
        return x
   
def init_weights(m):
    classname=m.__class__.__name__
    if classname.find('Conv')!=-1:
        m.weight.data.normal_(0.0,0.002)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            this_obs=[]
            this_act=[]
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                this_obs.append(obs)
                this_act.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)
            observations.append(this_obs)
            actions.append(this_act)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        print( (np.array(observations)).shape)
        print( (np.array(actions)).shape)
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
    
    #train the network
    o_expert=expert_data['observations']
    (N,N_step,N_obs)=o_expert.shape
    a_expert=expert_data['actions']
    (N,N_step,_,N_action)=a_expert.shape
    net=CNN(N_obs, N_action)
    
    #todo:initilize network parameters
    net.apply(init_weights)

    import torch.optim as optim
    optimizer=optim.Adam(net.parameters(),lr=1e-3, weight_decay=5e-12)
    criterion=nn.MSELoss()
    loss_history=[]
    for j in range(args.num_epochs):
        print("epoch %i"%j)
        (N,N_step,N_obs)=o_expert.shape
        (N,N_step,_,N_action)=a_expert.shape
        for k in range(max_steps):
            index=k
            o=Variable(torch.from_numpy(o_expert[:,index,:]).reshape(N,1,N_obs))
            o=o.float()
            a_out=net.forward(o)
            a_label=torch.from_numpy(a_expert[:,index,:].reshape(N,N_action,1))
            loss=criterion(a_out.float(), a_label.float())
            loss.backward()
            loss_history.append(loss)
            optimizer.step()
        print("before DAGGER")
        print(loss) 

        #implement dagger
        with tf.Session():
            tf_util.initialize()

            import gym
            env = gym.make(args.envname)
            max_steps = args.max_timesteps or env.spec.timestep_limit

            o_new_expert=[]
            a_new_expert=[]
            for i in range (int(args.num_rollouts)//2):
                this_o_new=[]
                this_a_new=[]
                obs=env.reset()
                done=False
                steps=0
                while not done:
                    action = policy_fn(obs[None, :])
                    this_o_new.append(obs)
                    this_a_new.append(action)
                    obs=Variable(torch.Tensor(obs).reshape(1,1,N_obs))
                    action_new=net.forward(obs).detach().numpy()
                    obs,r,done,_=env.step(action_new.reshape(17))
                    steps+=1
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break

                #if terminates early, we pad 0 to both observation and actions lists
                while steps<max_steps:
                    steps+=1
                    this_o_new.append(np.zeros(N_obs))
                    this_a_new.append(np.zeros((1,N_action)))
                o_new_expert.append(this_o_new)
                a_new_expert.append(this_a_new)
            o_new=np.array(o_new_expert)
            a_new=np.array(a_new_expert)
            o_expert=np.concatenate((o_expert,o_new), axis=0)
            a_expert=np.concatenate((a_expert,a_new), axis=0)
        
    plt.plot(loss_history, '-o')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig('/Users/joker/imitation_learning/humanoid_dagger.png')
    plt.show()

main()
