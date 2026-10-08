import sys
from src.rwd_func import *
from src.train import *
import torch

seed = int(sys.argv[1])
torch.manual_seed(seed)

num_iterations = 512
num_epochs = 5
num_transitions = 200
minibatch_size = 20

lr = 7.379879632415984e-05
lr_gamma = 0.32460358795404193

pol_clip_epsilon = 0.26692209754627677
val_clip_epsilon = 0.18639443443962206

lr_upd_freq = 5     # Should really be called period but I'm no physicist so it doesn't matter ;) 
gae_gamma = 0.516170793570394
lmbda = 0.23688335474478636
entropy_coef =  0.2659486523965797

max_charge = 5
max_steps = 25

rwd = abs_tot_err_rwd
log_file = open(sys.argv[2], "w")

agent = init_agent(num_epochs, num_transitions, minibatch_size, lr, lr_gamma,
                   gae_gamma, lmbda, pol_clip_epsilon, val_clip_epsilon, entropy_coef,
                   max_charge)
env = init_env(max_charge, max_steps, rwd, seed)

pol_losses, val_losses, tot_losses = train(num_iterations, num_transitions, num_epochs,
                                              log_file, lr_upd_freq, agent, env)[:3]

plot(num_iterations, pol_losses, val_losses, tot_losses)