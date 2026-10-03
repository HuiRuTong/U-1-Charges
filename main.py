from src.rwd_func import *
from src.train import *
import torch

num_iterations = 512
num_epochs = 5
num_transitions = 200
minibatch_size = 20

lr = 5.90690491231212e-05
lr_gamma = 0.24679173960802184

pol_clip_epsilon = 0.2261475443703728
val_clip_epsilon = 0.8173139210436976

lr_upd_freq = 5     # Should really be called period but I'm no physicist so it doesn't matter ;) 
gamma = 0.27492982926816606
lmbda = 0.7432675671334763
entropy_coef = 0.4245575317966134

max_charge = 5
max_steps = 25

rwd = abs_err_rwd
log_file = open("./found_charges/abs_1.txt", "w")

agent = init_agent(num_epochs, num_transitions, minibatch_size, lr, lr_gamma,
                   pol_clip_epsilon, val_clip_epsilon, gamma, lmbda, entropy_coef,
                   max_charge)
env = init_env(max_charge, max_steps, rwd)

policy_losses, val_losses, tot_losses = train(num_iterations, num_transitions, num_epochs,
                                              log_file, lr_upd_freq, agent, env)[:3]

plot(num_iterations, policy_losses, val_losses, tot_losses)