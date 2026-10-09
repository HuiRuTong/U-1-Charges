import argparse
from src.rwd_func import *
from src.train import *
import torch

rwd_funcs = {"abs_err_rwd":abs_err_rwd, "abs_tot_err_rwd":abs_tot_err_rwd}

parser = argparse.ArgumentParser()
argnames = {"num_iterations":int, "num_epochs":int, "num_transitions":int, "minibatch_size":int,
            "lr":float, "lr_gamma":float, "pol_clip_epsilon":float, "val_clip_epsilon":float, "lr_upd_freq":int,
            "gae_gamma":float, "lmbda":float, "entropy_coef":float, "max_charge":int, "max_steps":int,
            "rwd":str, "--seed":int, "--log":str, "--fig":str}
for arg, dtype in argnames.items():
    parser.add_argument(arg, type=dtype)
args = parser.parse_args()

seed = int(args.seed)
torch.manual_seed(seed)

rwd = rwd_funcs[args.rwd]
log_file = open(args.log, "w")

agent = init_agent(args.num_epochs, args.num_transitions, args.minibatch_size, args.lr, args.lr_gamma,
                   args.gae_gamma, args.lmbda, args.pol_clip_epsilon, args.val_clip_epsilon, args.entropy_coef,
                   args.max_charge)
env = init_env(args.max_charge, args.max_steps, rwd, seed)

num_solutions, pol_losses, val_losses, tot_losses = train(args.num_iterations, args.num_transitions, args.num_epochs,
                                                          log_file, args.lr_upd_freq, agent, env)[:4]

plot(args.fig, args.num_iterations, num_solutions, pol_losses, val_losses, tot_losses)