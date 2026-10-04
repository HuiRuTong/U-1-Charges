from src.rwd_func import *
from src.train import *
import torch
import optuna

rwd_funcs = {"abs_err_rwd":abs_err_rwd, "abs_tot_err_rwd":abs_tot_err_rwd}

def objective(trial):
    num_iterations = 256
    num_epochs = 5
    num_transitions = 200
    minibatch_size = 20

    lr = trial.suggest_float("lr", 1e-8, 1e-4, log=True)
    lr_gamma = trial.suggest_float("lr_gamma", 0.1, 0.5, log=True)

    pol_clip_epsilon = trial.suggest_float("pol_clip_epsilon", 0.1, 0.9, log=True)
    val_clip_epsilon = trial.suggest_float("val_clip_epsilon", 0.1, 0.9, log=True)

    lr_upd_freq = 5     # Should really be called period but I'm no physicist so it doesn't matter ;) 
    gae_gamma = trial.suggest_float("gamma", 0.1, 0.9, log=True)
    lmbda = trial.suggest_float("lmbda", 0.1, 0.9, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 0.1, 0.9, log=True)

    max_charge = 5
    max_steps = 25
    
    rwd = rwd_funcs[trial.suggest_categorical("rwd_func", ["abs_err_rwd", "abs_tot_err_rwd"])]
    log_file = None

    agent = init_agent(num_epochs, num_transitions, minibatch_size, lr, lr_gamma,
                       gae_gamma, lmbda, pol_clip_epsilon, val_clip_epsilon, entropy_coef,
                       max_charge)
    env = init_env(max_charge, max_steps, rwd)

    return train(num_iterations, num_transitions, num_epochs,
                 log_file, lr_upd_freq, agent, env, trial)[3]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

print("Study statistics: ")
print("\tNumber of finished trials: ", len(study.trials))
print("\tNumber of pruned trials: ", len(pruned_trials))
print("\tNumber of complete trials: ", len(complete_trials))

print("Best trial:")
print("\tValue: ", study.best_trial.value)
print("\tParams: ")
for key, value in study.best_trial.params.items():
    print(f"\t{key}: {value}")