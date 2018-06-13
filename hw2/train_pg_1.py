#Reference: 
#1. https://github.com/mabirck/CS294-DeepRL/blob/master/lectures/class-5/REINFORCE.py
#2. https://github.com/JamesChuanggg/pytorch-REINFORCE/blob/master/reinforce_continuous.py
#3. https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
# With the help from the implementations above, I was finally able to translate the provided skeleton code in Tensorflow into the code below

import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.init as ini
import random

#============================================================================================#
# Utilities
#============================================================================================#
  
class Policy_discrete(nn.Module):
    def __init__(self, inputdim, outputdim, n_layers, hiddendim, activation, output_activation):
        super(Policy_discrete, self).__init__()
        if (output_activation==None):
            self.original_output=True
        else:
            self.original_output=False
        self.history_of_log_probs=[]
        self.layers=nn.ModuleList()
        for i in range(n_layers):
            if(i==0):
                self.layers.append(nn.Linear(inputdim, hiddendim))
                self.layers.append(activation)
            elif(i==(n_layers-1)):
                self.layers.append(nn.Linear(hiddendim, outputdim))
                if(output_activation!=None):
                    self.layers.append(output_activation)
            else:
                self.layers.append(nn.Linear(hiddendim, hiddendim))
                self.layers.append(activation)
    def forward(self, x):
        for i, l in enumerate(self.layers):
            x=l(x)
        return x
    def run(self, x):
        x=Variable(Tensor(x))
        p=self(x)
        if self.original_output:
            d=Categorical(logits=p)
        else:
            #Suppose after the output_activation, we get the probability(i.e. a softmax activation)
            #This assumption might be false.
            d=Categorical(probs=p)
        action=d.sample()
        self.history_of_log_probs.append(d.log_prob(action))
        return action #haven't checked the type of action, might be buggy here
    def learn(self, optimizer, history_of_rewards, gamma, reward_to_go):
        total_weighted_reward=Variable(torch.zeros(1,1))
        gradient=Variable(torch.zeros(1,1))
        loss=0
        if !reward_to_go:
            #sum up all the reward along the trajectory
            for i in reversed(range(len(history_of_rewards))):
                total_weighted_reward = gamma * total_weighted_reward + rewards[i]
                gradient+=self.history_of_log_probs[i]
            loss=loss-gradient*total_weighted_reward
            loss=loss/len(history_of_rewards) #in case the episode terminates early
        else:
            #reward to go mode
            for i in reversed(range(len(history_of_rewards))):
                total_weighted_reward=gamma*total_weighted_reward+rewards[i]
                loss=loss-self.history_of_log_probs[i]*total_weighted_reward
            loss=loss/len(history_of_rewards) #in case the episode terminates early
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.history_of_log_probs=[]
                
        
        
class Policy_continuous_hw(nn.Module): #this policy network only outputs the mean of the Gaussian 
    def __init__(self, inputdim, outputdim, n_layers, hiddendim, activation, output_activation):
        super(Policy_continuous_mean_only, self).__init__()
        if (output_activation==None):
            self.original_output=True
        else:
            self.original_output=False
        self.history_of_log_probs=[]
        self.logstd_raw=nn.Parameter(torch.ones(outputdim), requires_grad=True)
        self.outputid=Variable(torch.eyes(outputdim), requires_grad=False)
        self.layers=nn.ModuleList()
        for i in range(n_layers):
            if(i==0):
                self.layers.append(nn.Linear(inputdim, hiddendim))
                self.layers.append(activation)
            elif(i==(n_layers-1)):
                self.layers.append(nn.Linear(hiddendim, outputdim))
                if(output_activation!=None):
                    self.layers.append(output_activation)
            else:
                self.layers.append(nn.Linear(hiddendim, hiddendim))
                self.layers.append(activation)
    def forward(self, x):
        for i, l in enumerate(self.layers):
            x=l(x)
        return x
    def run(self, x):
        x=Variable(Tensor(x))
        #the action space is continuous
        u=self(x)
        sigma2=torch.exp(self.logstd_raw)*self.outputid
        d=MultivariateNormal(u, sigma2)
        action=d.sample()
        self.history_of_log_probs.append(d.log_prob(action))
        return action
    def learn(self, optimizer, history_of_rewards, gamma, reward_to_go):
        total_weighted_reward=Variable(torch.zeros(1,1))
        gradient=Variable(torch.zeros(1,1))
        loss=0
        if !reward_to_go:
            #sum up all the reward along the trajectory
            for i in reversed(range(len(history_of_rewards))):
                total_weighted_reward = gamma * total_weighted_reward + rewards[i]
                gradient+=self.history_of_log_probs[i]
            loss=loss-(gradient*total_weighted_reward.expand(gradient.size())).sum()
            loss=loss/len(history_of_rewards) #in case the episode terminates early
        else:
            #reward to go mode
            for i in reversed(range(len(history_of_rewards))):
                total_weighted_reward=gamma*total_weighted_reward+rewards[i]
                loss=loss-(self.history_of_log_probs[i]*total_weighted_reward.expand(self.history_of_log_probs[i].size())).sum()
            loss=loss/len(history_of_rewards) #in case the episode terminates early
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.history_of_log_probs=[]
        
class Critic(nn.Module): #Critic is always discrete
    def __init__(self, inputdim, outputdim, n_layers, hiddendim, activation, output_activation):
        super(Critic, self).__init__()
        if (output_activation==None):
            self.original_output=True
        else:
            self.original_output=False
        self.history_of_values=[]
        self.layers=nn.ModuleList()
        for i in range(n_layers):
            if(i==0):
                self.layers.append(nn.Linear(inputdim, hiddendim))
                self.layers.append(activation)
            elif(i==(n_layers-1)):
                self.layers.append(nn.Linear(hiddendim, outputdim))
                if(output_activation!=None):
                    self.layers.append(output_activation)
            else:
                self.layers.append(nn.Linear(hiddendim, hiddendim))
                self.layers.append(activation)
    def forward(self, x):
        for i, l in enumerate(self.layers):
            x=l(x)
        return x
    def run(self, x):
        x=Variable(Tensor(x))
        v=self(x)
        self.history_of_values.append(v)
        return v #haven't checked the type of value, might be buggy here
    def learn(self, optimizer, history_of_rewards, gamma):
        total_weighted_reward=0
        gradient=Variable(torch.zeros(1,1))
        loss=0
        history_of_total_weighted_reward=[]
        for i in reversed(range(len(history_of_rewards))):
            total_weighted_reward=gamma*total_weighted_reward+rewards[i]
            history_of_total_weighted_reward.insert(0,total_weighted_reward)
        history_of_total_weighted_reward=torch.tensor(history_of_total_weighted_reward)
        #rescale the reward value(do not want to compute raw Q value)
        reward_u=history_of_total_weighted_reward.mean()
        reward_std=history_of_total_weighted_reward.std()+1e-8
        history_of_total_weighted_reward=(history_of_total_weighted_reward-reward_u)/reward_std
        for i in range(len(self.history_of_values)):
            loss+=F.mse_loss(history_of_values[i], history_of_weighted_reward[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.history_of_values=[]

class Agent(nn.Module):
    def __init__(gamma=1.0,min_timesteps_per_batch=1000,max_path_length=None,learning_rate=5e-3,\ 
             reward_to_go=True, normalize_advantages=True,nn_baseline=False,n_layers=1,size=32):
        self.actor=build_mlp(ob_dim, ac_dim, "actor", n_layers=n_layers, size=size)
        self.actor_optimizer=optim.Adam(actor.parameters(), lr=learning_rate)
        if nn_baseline:
            self.critic = build_mlp(ob_dim,1,"nn_baseline", n_layers=n_layers,size=size)
            self.critic_optimizer=
    
def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=torch.nn.Tanh,
        output_activation=None,
        discrete=True
        ):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units. 
    # 
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#
    if scope="nn_baseline":
        print("critic activated.")
        return Critic(input_placeholder, output_size, n_layers, size, activation, output_activation) #Critic is always discrete
    else:
        #return an actor
        if discrete:
            print("discrete-type actor activated.")
            return Policy_discrete(input_placeholder, output_size, n_layers, size, activation, output_activation)
        else:
            print("continuous-type actor activated.")
            return Policy_continuous_hw(input_placeholder, output_size, n_layers, size, activation, output_activation)

def pathlength(path):
    return len(path["reward"])



#============================================================================================#
# Policy Gradient
#============================================================================================#

def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100, 
             gamma=1.0, 
             min_timesteps_per_batch=1000, 
             max_path_length=None,
             learning_rate=5e-3, 
             reward_to_go=True, 
             animate=True, 
             logdir=None, 
             normalize_advantages=True,
             nn_baseline=False, 
             seed=0,
             # network arguments
             n_layers=1,
             size=32
             ):

    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)

    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    #create actor
    actor=build_mlp(ob_dim, ac_dim, "actor", n_layers=n_layers, size=size)
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-3)

    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#

    if nn_baseline:
        critic = build_mlp(ob_dim, 
                                1, 
                               "nn_baseline",
                                n_layers=n_layers,
                                size=size)
        critic_optimizer=optim.Adam(critic.parameters(), lr=3e-3)
    
    #todo: initilize actor and critic

    #========================================================================================#
    # Tensorflow Engineering: Config, Session, Variable initialization
    #========================================================================================#

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 

    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`



    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0

    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                obs.append(ob)
                ac = actor.run(ob)
                print("need to type-check action here:(two lines)")
                print(ac)
                print(ac.size())
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            #One episode finishes; perform update here
            finish_episode(actor, actor_optimizer, critic=None, critic_optimizer=None, )
            path = {"observation" : np.array(obs), 
                    "reward" : np.array(rewards), 
                    "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch



        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()


    
    