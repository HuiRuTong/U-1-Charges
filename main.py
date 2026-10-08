import sys
from src.rwd_func import *
from src.train import *
import torch

seed = int(sys.argv[1])
torch.manual_seed(seed)

num_iterations = 512
num_transitions = 200
num_epochs = 5
minibatch_size = 20

actor_lr = 6.010731706199868e-05
critic_lr = 2.0150622953681204e-07
actor_lr_gamma = 0.11737292818213454
critic_lr_gamma = 0.17496431209947724
actor_clip_epsilon = 0.5737968512498365
critic_clip_epsilon = 0.7325469386729884

lr_upd_freq = 5      # Should really be called period but I'm no physicist so it doesn't matter ;) 
gae_gamma = 0.2953513202609301
lmbda = 0.5738938980970808
entropy_coef = 0.6581407850467246

max_charge = 6
max_steps = 25

rwd = abs_err_rwd
log_file = open(sys.argv[2], "w")

agent = init_agent(num_epochs, num_transitions, minibatch_size, max_charge, actor_lr, critic_lr,
                   actor_lr_gamma, critic_lr_gamma, actor_clip_epsilon, critic_clip_epsilon,
                   gae_gamma, lmbda, entropy_coef)
env = init_env(max_charge, max_steps, rwd, seed)

pol_losses, val_losses = train(num_iterations, num_transitions, num_epochs, log_file, lr_upd_freq, agent, env)[:2]

plot(num_iterations, pol_losses, val_losses)