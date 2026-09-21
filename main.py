import numpy as np
from src.neural_net import *
from src.rwd_func import *
import matplotlib.pyplot as plt
import torch

num_iterations = 512
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

agent = PPO(num_transitions, num_epochs, minibatch_size, max_charge, actor_lr, critic_lr,
            actor_lr_gamma, critic_lr_gamma, actor_clip_epsilon, critic_clip_epsilon,
            gae_gamma, lmbda, entropy_coef)
env = Charge_Env(max_charge, max_steps, abs_tot_err_rwd)

pol_losses = []
val_losses = []
tot_losses = []
found_charges = []


log_file = open("./found_charges/abs_2.txt", "w")

for i in range(num_iterations):

    print(f"Currently on iteration {i+1} of {num_iterations}")

    states = []
    actions = []
    log_probs = []
    entropies = []
    rewards = []
    ended = []
    vals = []
    vals_offset = []

    end_count = 0
    for j in range(num_transitions):
        state = torch.tensor(env.charges, dtype=torch.float32)
        states.append(state)

        action, log_prob = agent.get_action(torch.unsqueeze(state, 0))
        actions.append(action)
        log_probs.append(log_prob)

        val = torch.flatten(agent.critic(torch.unsqueeze(state, 0)))
        vals.append(val)
        vals_offset.append(end_count)

        state, reward, terminated, truncated, info = env.step(action, found_charges, log_file)
        rewards.append(reward)

        ended.append(int(terminated or truncated))

        if ended[-1] or j == num_transitions - 1:  # Last state also requires the next value
            state = torch.tensor(state, dtype=torch.float32)
            vals.append(int(not terminated) * agent.critic(torch.unsqueeze(state, 0)))   # Terminated states will hvae zero value

            end_count += 1

            env.reset()
    
    print(f"Mean rewards: {env.rewards_sum / num_transitions: .2f}")
    env.rewards_sum = 0.0

    agent.states = torch.stack(states).detach()
    agent.actions = torch.stack(actions).detach()
    agent.log_probs = torch.stack(log_probs).detach()
    agent.vals = torch.stack(vals).detach()
    agent.vals_offset = torch.tensor(vals_offset)
    agent.rewards = (torch.tensor(rewards))
    agent.ended = torch.tensor(ended)

    # agent.rewards = ((agent.rewards - torch.mean(agent.rewards)) / torch.std(agent.rewards)).detach()
    agent.calc_gae_tar()

    batch_pol_loss = 0
    batch_val_loss = 0
    for j in range(num_epochs):
        print(f"\t Epoch {j+1} of {num_epochs}", end='')

        batch_pol_loss_, batch_val_loss_ = agent.upd(torch.randperm(num_transitions))
        print(f"\t policy loss: {batch_pol_loss_: .2f}, val loss: {batch_val_loss_: .2f}")

        """if num_epochs // (j+1) == lr_upd_freq:
            agent.actor_scheduler.step()
            agent.critic_scheduler.step()"""

        batch_pol_loss += batch_pol_loss_
        batch_val_loss += batch_val_loss_

    pol_losses.append((batch_pol_loss / num_epochs).detach())
    val_losses.append((batch_val_loss / num_epochs).detach())

    print(f"End of iteration\nNumber of solutions found so far: {len(found_charges)}")

log_file.close()

fig, axs = plt.subplots(1, 2)
itierations = np.arange(1, num_iterations+1)
axs[0].plot(itierations, pol_losses, label="policy loss", color="r")
axs[1].plot(itierations, val_losses, label="val loss", color="b")
plt.show()