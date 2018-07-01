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
import torch.distributions as D
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
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
        self.activation=activation
        self.output_activation=output_activation
        self.n_layers=n_layers+1
        if self.n_layers==1:
            self.layers.append(nn.Linear(inputdim, outputdim))
        else:
            for i in range(self.n_layers):
                if(i==0):
                    self.layers.append(nn.Linear(inputdim, hiddendim))
                elif(i==(self.n_layers-1)):
                    self.layers.append(nn.Linear(hiddendim, outputdim))
                else:
                    self.layers.append(nn.Linear(hiddendim, hiddendim))
    def forward(self, x):
        for i, l in enumerate(self.layers):
            if (i<(self.n_layers-1)):
                x=l(x)
                x=self.activation(x)
            else:
                x=l(x)
                if self.original_output:
                    return x
                else:
                    x=self.output_activation(x)
                    return x
    def run(self, x):
        x=Variable(x)
        p=self(x)
        if self.original_output:
            d=Categorical(logits=p)
        else:
            #Suppose after the output_activation, we get the probability(i.e. a softmax activation)
            #This assumption might be false.
            d=Categorical(probs=p)
        action=d.sample()
        log_prob=d.log_prob(action)
        return action, log_prob 

class Policy_continuous(nn.Module):
    def __init__(self, inputdim, outputdim, n_layers, hiddendim, activation, output_activation):
        super(Policy_continuous, self).__init__()
        if (output_activation==None):
            self.original_output=True
        else:
            self.original_output=False
        self.activation=activation
        self.output_activation=output_activation
        self.history_of_log_probs=[]
        self.n_layers=n_layers+1
        self.layers=nn.ModuleList()
        if self.n_layers==1:
            self.mean=nn.Linear(inputdim, outputdim)
            self.logstd_raw=nn.Linear(inputdim, outputdim)
        else:
            for i in range(self.n_layers-1):
                if(i==0):
                    self.layers.append(nn.Linear(inputdim, hiddendim))
                else:
                    self.layers.append(nn.Linear(hiddendim, hiddendim))
            self.mean=nn.Linear(hiddendim, outputdim)
            self.logstd_raw=nn.Linear(hiddendim, outputdim)
    def forward(self, x):
        for i, l in enumerate(self.layers):
            x=l(x)
            x=self.activation(x)
        u=self.mean(x)
        logstd=self.logstd_raw(x)
        if self.original_output:
            return u, logstd
        else:
            u=self.output_activation(u)
            logstd=self.output_activation(logstd)
            return u, logstd
    def run(self, x):
        x=Variable(x)
        u, logstd=self(x)
        d=D.Normal(loc=u, scale=logstd.exp()) #might want to use N Gaussian instead
        action=d.sample().detach()
        log_prob=d.log_prob(action).sum(1).view(-1,1)
        return action, log_prob
        
class Critic(nn.Module): #Critic is always discrete
    def __init__(self, inputdim, outputdim, n_layers, hiddendim, activation, output_activation):
        super(Critic, self).__init__()
        if (output_activation==None):
            self.original_output=True
        else:
            self.original_output=False
        self.history_of_values=[]
        self.layers=nn.ModuleList()
        self.activation=activation
        self.output_activation=output_activation
        self.n_layers=n_layers+1
        if self.n_layers==1:
            self.layers.append(nn.Linear(inputdim, outputdim))
        else:
            for i in range(self.n_layers):
                if(i==0):
                    self.layers.append(nn.Linear(inputdim, hiddendim))
                elif(i==(self.n_layers-1)):
                    self.layers.append(nn.Linear(hiddendim, outputdim))
                else:
                    self.layers.append(nn.Linear(hiddendim, hiddendim))
    def forward(self, x):
        for i, l in enumerate(self.layers):
            if (i<(self.n_layers-1)):
                x=l(x)
                x=self.activation(x)
            else:
                x=l(x)
                if self.original_output:
                    return x
                else:
                    x=self.output_activation(x)
                    return x
    def run(self, x):
        x=Variable(x)
        v=self(x)
        return v 

def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=torch.nn.functional.tanh,
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
    if scope=="nn_baseline":
        print("critic activated.")
        return Critic(input_placeholder, output_size, n_layers, size, activation, output_activation) 
        #(Note that Critic is always discrete)
    else:
        #return an actor
        if discrete:
            print("discrete-type actor activated.")
            return Policy_discrete(input_placeholder, output_size, n_layers, size, activation, output_activation)
        else:
            print("continuous-type actor activated.")
            return Policy_continuous(input_placeholder, output_size, n_layers, size, activation, output_activation)

def pathlength(path):
    return len(path["reward"])

def reinforce_loss(log_prob, a, num_path):
    return - (log_prob.view(-1, 1) * a).sum() / num_path

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
             size=32,
             network_activation='tanh',
             output_activation='None'
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
    env.seed(seed)    
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
    
    #activation function for the network
    if network_activation=='relu':
        activation=torch.nn.functional.relu
    elif network_activation=='leaky_relu':
        activation=torch.nn.functional.leaky_relu
    else:
        activation=torch.nn.functional.tanh
        
    #output activation function for the network
    if output_activation=='relu':
        output_a=torch.nn.functional.relu
    elif output_activation=='leaky_relu':
        output_a=torch.nn.functional.leaky_relu
    elif output_activation=='tanh':
        output_a=torch.nn.functional.tanh
    else:
        output_a=None
    
    #create policy
    actor=build_mlp(ob_dim, ac_dim, "actor",\
                    n_layers=n_layers, size=size, activation=activation,\
                    output_activation=output_a, discrete=discrete)
    actor_loss=reinforce_loss
    actor_optimizer=torch.optim.Adam(actor.parameters(), lr=learning_rate)
    
    #========================================================================================#
    #                           ----------SECTION 5----------
    # Optional Baseline
    #========================================================================================#
    if nn_baseline:
        critic=build_mlp(ob_dim,1, "nn_baseline", n_layers=n_layers, size=size,\
                         activation=activation, output_activation=torch.nn.functional.tanh, \
                         discrete=discrete)
        critic_loss=nn.MSELoss()
        critic_optimizer=torch.optim.Adam(critic.parameters(), lr=learning_rate)
        

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
            obs, acs, rewards, log_probs = [], [], [], []
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                ob = torch.from_numpy(ob).float().unsqueeze(0)
                obs.append(ob)
                ac, log_prob = actor.run(ob)
                acs.append(ac)
                log_probs.append(log_prob)
                #format the action from policy
                if discrete:
                    ac = int(ac)
                else:
                    ac = ac.squeeze(0).numpy()
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {"observation" : torch.cat(obs, 0),
                    "reward" : torch.Tensor(rewards),
                    "action" : torch.cat(acs, 0),
                    "log_prob" : torch.cat(log_probs, 0)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        ob_no = torch.cat([path["observation"] for path in paths], 0)
        ac_na = torch.cat([path["action"] for path in paths], 0)
                                   
        #====================================================================================#
        #                           ----------SECTION 4----------
        # Computing Q-values
        #
        # Your code should construct numpy arrays for Q-values which will be used to compute
        # advantages (which will in turn be fed to the placeholder you defined above). 
        #
        # Recall that the expression for the policy gradient PG is
        #
        #       PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
        #
        # where 
        #
        #       tau=(s_0, a_0, ...) is a trajectory,
        #       Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
        #       and b_t is a baseline which may depend on s_t. 
        #
        # You will write code for two cases, controlled by the flag 'reward_to_go':
        #
        #   Case 1: trajectory-based PG 
        #
        #       (reward_to_go = False)
        #
        #       Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
        #       entire trajectory (regardless of which time step the Q-value should be for). 
        #
        #       For this case, the policy gradient estimator is
        #
        #           E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
        #
        #       where
        #
        #           Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
        #
        #       Thus, you should compute
        #
        #           Q_t = Ret(tau)
        #
        #   Case 2: reward-to-go PG 
        #
        #       (reward_to_go = True)
        #
        #       Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
        #       from time step t. Thus, you should compute
        #
        #           Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        #
        #
        # Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
        # like the 'ob_no' and 'ac_na' above. 
        #
        #====================================================================================#
        q_n = []
        for path in paths:
            rewards = path['reward']
            num_steps = pathlength(path)
            R=[]
            if reward_to_go:
                for t in range(num_steps):
                    R.append((torch.pow(gamma, torch.arange(num_steps-t))*rewards[t:]).sum().view(-1,1))
                q_n.append(torch.cat(R))
            else:
                q_n.append((torch.pow(gamma, torch.arange(num_steps)) * rewards).sum() * torch.ones(num_steps, 1))
        q_n = torch.cat(q_n, 0)
        
         #====================================================================================#
        #                           ----------SECTION 5----------
        # Computing Baselines
        #====================================================================================#
        if nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
            # #bl2 below.)
            b_n = critic(ob_no)
            q_n_std = q_n.std()
            q_n_mean = q_n.mean()
            b_n_scaled = b_n * q_n_std + q_n_mean
            adv_n = (q_n - b_n_scaled).detach()
        else:
            adv_n = q_n
        #====================================================================================#
        #                           ----------SECTION 4----------
        # Advantage Normalization
        #====================================================================================#

        if normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1. 
            # YOUR_CODE_HERE
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + np.finfo(np.float32).eps.item())
        
        #====================================================================================#
        #                           ----------SECTION 5----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if nn_baseline:
            # ----------SECTION 5----------
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

            # YOUR_CODE_HERE
            target = (q_n - q_n_mean) / (q_n_std + np.finfo(np.float32).eps.item())
            critic_optimizer.zero_grad()
            c_loss = critic_loss(b_n, target)
            c_loss.backward()
            critic_optimizer.step()
            
        #====================================================================================#
        #                           ----------SECTION 4----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR_CODE_HERE
        log_probs = torch.cat([path["log_prob"] for path in paths], 0)
        actor_optimizer.zero_grad()
        loss = actor_loss(log_probs, adv_n, len(paths))
        print(loss)
        loss.backward()
        actor_optimizer.step()

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
    parser.add_argument('--activation', '-a', type=str, default='tanh')
    parser.add_argument('--output_activation', '-oa', type=str, default='None')
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
                size=args.size,
                network_activation=args.activation,
                output_activation=args.output_activation
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
        

if __name__ == "__main__":
    main()


    
    
