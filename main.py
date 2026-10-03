from src.rwd_func import *
from src.train import *
import torch

num_iterations = 20
num_transitions = 200
num_epochs = 5
minibatch_size = 20

actor_lr = 1e-5
critic_lr = 2.5e-6
actor_lr_gamma = 0.2
critic_lr_gamma = 0.1
actor_clip_epsilon = 0.2
critic_clip_epsilon = 0.2

lr_upd_freq = 5      # Should really be called period but I'm no physicist so it doesn't matter ;) 
gae_gamma = 0.85
lmbda = 0.5
entropy_coef = 0.5

max_charge = 6
max_steps = 25

rwd = abs_err_rwd
log_file = open("./found_charges/abs_2.txt", "w")

agent = init_agent(num_epochs, num_transitions, minibatch_size, max_charge, actor_lr, critic_lr,
                   actor_lr_gamma, critic_lr_gamma, actor_clip_epsilon, critic_clip_epsilon,
                   gae_gamma, lmbda, entropy_coef)
env = init_env(max_charge, max_steps, rwd)

pol_losses, val_losses = train(num_iterations, num_transitions, num_epochs, log_file, lr_upd_freq, agent, env)[:2]

plot(num_iterations, pol_losses, val_losses)