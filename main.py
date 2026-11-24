import argparse
from src.rwd_func import *
from src.train import *
import torch

# 1000 5 200 20 1e-4 1 0.49 0.33 5 0.95 0.37 0.14 5 150 min_err_rwd --seed 0 --log ./test.txt --fig ./test.png

rwd_funcs = {"abs_err_rwd":abs_err_rwd, "abs_tot_err_rwd":abs_tot_err_rwd, "min_err_rwd":min_err_rwd,
             "potential_rwd":potential_rwd}

parser = argparse.ArgumentParser()
argnames = {"num_iterations":int, "num_epochs":int, "num_transitions":int, "minibatch_size":int,
            "lr":float, "lr_gamma":float, "pol_clip_epsilon":float, "val_clip_epsilon":float, "lr_upd_freq":int,
            "gae_gamma":float, "lmbda":float, "entropy_coef":float, "max_charge":int, "max_steps":int,
            "rwd":str, "--seed":int, "--log":str, "--fig":str}
for arg, dtype in argnames.items():
    parser.add_argument(arg, type=dtype)
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

rwd = rwd_funcs[args.rwd]
log_file = open(args.log, "w")

agent = init_agent(args.num_epochs, args.num_transitions, args.minibatch_size, args.lr, args.lr_gamma,
                   args.gae_gamma, args.lmbda, args.pol_clip_epsilon, args.val_clip_epsilon, args.entropy_coef,
                   args.max_charge)
env = init_env(args.max_charge, args.max_steps, rwd, args.gae_gamma, args.seed)

num_solutions, pol_losses, val_losses, tot_losses, exp_vars, entropies = \
train(args.num_iterations, args.num_transitions, args.num_epochs, args.max_steps, log_file, args.lr_upd_freq, agent, env)

plot_1(args.fig, args.num_iterations, num_solutions, tot_losses, exp_vars, entropies)