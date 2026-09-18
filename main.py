import numpy as np
from src.neural_net import *
from src.rwd_func import *
import matplotlib.pyplot as plt
import torch

num_iterations = 512
num_transitions = 200
num_epochs = 5
minibatch_size = 20

lr = 1e-6
lr_gamma = 0.2

pol_clip_epsilon = 0.2
val_clip_epsilon = 0.2

lr_upd_freq = 5     # Should really be called period but I'm no physicist so it doesn't matter ;) 
gamma = 0.85
lmbda = 0.5
entropy_coef = 0.5

max_charge = 5
max_steps = 25

agent = PPO(num_epochs, num_transitions, minibatch_size, max_charge, lr,
            lr_gamma, gamma, lmbda, pol_clip_epsilon, val_clip_epsilon, entropy_coef)
env = Charge_Env(max_charge, max_steps, abs_tot_err_rwd)

policy_losses = []
val_losses = []
tot_losses = []
found_charges = []

log_file = open("./found_charges/abs_1.txt", "w")

for i in range(num_iterations):

    print(f"Currently on iteration {i+1} of {num_iterations}")

    states = []
    actions = []
    log_probs = []
    entropies = []
    rwd_func = []
    ended = []
    vals = []
    vals_offset = []

    end_count = 0
    for j in range(num_transitions):
        state = torch.tensor(env.charges, dtype=torch.float32)
        states.append(state)

        action, log_prob, val = agent.get_action_val(torch.unsqueeze(state, 0))
        actions.append(action)
        log_probs.append(log_prob)
        vals.append(val)
        vals_offset.append(end_count)

        state, reward, terminated, truncated, info = env.step(action, found_charges, log_file)
        rwd_func.append(reward)

        ended.append(int(terminated or truncated))

        if terminated or truncated or j == num_transitions - 1:  # Last non terminal / truncated state also requires the next value
            state = torch.tensor(state, dtype=torch.float32)
            vals.append(int(not terminated) * agent.get_action_val(torch.unsqueeze(state, 0), value_only=True))   # Terminated states will hvae zero value

            end_count += 1
            vals_offset.append(end_count)

            env.reset()

    print(f"Mean rewards: {env.rewards_sum / num_transitions: .2f}")
    env.rewards_sum = 0.0

    agent.states = torch.stack(states).detach()
    agent.actions = torch.stack(actions).detach()
    agent.log_probs = torch.stack(log_probs).detach()
    agent.vals = torch.stack(vals).detach()
    agent.vals_offset = torch.tensor(vals_offset)
    agent.rewards = torch.tensor(rwd_func)
    agent.ended = torch.tensor(ended)

    # agent.rewards = ((agent.rewards - torch.mean(agent.rewards)) / torch.std(agent.rewards)).detach()
    agent.calc_gae_tar()

    batch_pol_loss = 0
    batch_val_loss = 0
    batch_tot_loss = 0
    for j in range(num_epochs):
        print(f"\t Epoch {j+1} of {num_epochs}", end='')
    
        batch_pol_loss_, batch_val_loss_, batch_tot_loss_ = agent.upd(torch.randperm(num_transitions))
        print(f"\t policy loss: {batch_pol_loss_: .2f}, val loss: {batch_val_loss_: .2f}, tot loss: {batch_tot_loss_: .2f}")

        """if num_epochs // (j+1) == lr_upd_freq:
            agent.scheduler.step()"""

        batch_pol_loss += batch_pol_loss_
        batch_val_loss += batch_val_loss_
        batch_tot_loss += batch_tot_loss_
        
    policy_losses.append((batch_pol_loss / num_epochs).detach())
    val_losses.append((batch_val_loss / num_epochs).detach())
    tot_losses.append((batch_tot_loss / num_epochs).detach())

    print(f"End of itetration\nNumber of solutions found so far: {len(found_charges)}")

log_file.close()

fig, ax = plt.subplots(1, 1)
itierations = np.arange(1, num_iterations+1)
ax.plot(itierations, policy_losses, label="policy loss", color="r")
ax.plot(itierations, val_losses, label="val loss", color="b")
ax.plot(itierations, tot_losses, label="tot loss", color="m")

ax.legend(loc="upper right")
plt.show()
