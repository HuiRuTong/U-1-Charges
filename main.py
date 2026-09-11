import numpy as np
from src.neural_net import *
from src.rwd_func import *
import matplotlib.pyplot as plt
import torch

num_iterations = 1024
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

agent = PPO(num_epochs, num_transitions, minibatch_size, lr, lr_gamma, gamma, lmbda, pol_clip_epsilon, val_clip_epsilon, entropy_coef)
env = Charge_Env(max_charge, max_steps, abs_tot_err_rwd)

policy_losses = []
val_losses = []
tot_losses = []
found_charges = []

log_file = open("./found_charges/abs_3.txt", "w")

for i in range(num_iterations):

    print(f"Currently on iteration {i+1} of {num_iterations}")

    states = []
    actions = []
    log_probs = []
    entropies = []
    rwd_func = []
    ended = []
    vals = []

    for j in range(num_transitions):
        state = torch.tensor(env.charges, dtype=torch.float32)
        states.append(state)

        particle_logits, generation_logits, mod_logits, val = agent.actor_critic(torch.unsqueeze(state, 0))
        
        particle_distr = torch.distributions.Categorical(logits=particle_logits)
        chosen_particle = particle_distr.sample()
        particle_log_prob = particle_distr.log_prob(chosen_particle)

        if (chosen_particle.item() > 2):
            # To avoid picking 3rd charge for non doublet and neutrino
            generation_logits.masked_fill_(torch.tensor([False, False, True]), -torch.inf)

        generation_distr = torch.distributions.Categorical(logits=generation_logits)
        chosen_generation = generation_distr.sample()
        generation_log_prob = generation_distr.log_prob(chosen_generation)

        if (state[chosen_particle, chosen_generation.item()] <= -max_charge):
            mod_logits.masked_fill_(torch.tensor([True, False]), -torch.inf)
        elif (state[chosen_particle, chosen_generation.item()] >= max_charge):
            mod_logits.masked_fill_(torch.tensor([False, True]), -torch.inf)

        mod_distr = torch.distributions.Categorical(logits=mod_logits)
        chosen_mod = mod_distr.sample()
        mod_log_prob = mod_distr.log_prob(chosen_mod)

        action = torch.stack((chosen_particle, chosen_generation, chosen_mod))
        log_prob = torch.sum(torch.stack((particle_log_prob, generation_log_prob, mod_log_prob)), 0)
        val = torch.flatten(val)

        actions.append(action)
        log_probs.append(log_prob)
        vals.append(val)

        state, reward, terminated, truncated, info = env.step(action, found_charges, log_file)
        rwd_func.append(reward)

        ended.append(int(terminated or truncated))

        if terminated or truncated or j == num_transitions - 1:  # Last non terminal / truncated state also requires the next value
            state = torch.tensor(state, dtype=torch.float32)
            vals.append(int(not terminated) * agent.actor_critic.forward(torch.unsqueeze(state, 0), value_only=True))   # Terminated states will hvae zero value

            env.reset()

    print(f"Mean rewards: {env.rewards_sum / num_transitions}")
    env.rewards_sum = 0.0

    agent.states = torch.stack(states).detach()
    agent.actions = torch.stack(actions).detach()
    agent.log_probs = torch.stack(log_probs).detach()
    agent.vals = torch.stack(vals).detach()
    agent.rewards = torch.tensor(rwd_func)
    agent.ended = torch.tensor(ended)

    agent.rewards = ((agent.rewards - torch.mean(agent.rewards)) / torch.std(agent.rewards)).detach()
    agent.calc_gae_tar()

    policy_loss = 0
    val_loss = 0
    tot_loss = 0
    for j in range(num_epochs):
        print(f"\t Epoch {j+1} of {num_epochs}", end='')

        policy_loss_, val_loss_, tot_loss_ = agent.upd(torch.randperm(num_transitions))
        policy_loss += policy_loss_
        val_loss += val_loss_
        tot_loss += tot_loss_
        print(f"\t policy loss: {policy_loss: .2f}, val loss: {val_loss: .2f}, tot loss: {tot_loss: .2f}")

        if num_epochs // (j+1) == lr_upd_freq:
            agent.scheduler.step()

    policy_losses.append((policy_loss / num_epochs).item())
    val_losses.append((val_loss / num_epochs).item())
    tot_losses.append((tot_loss / num_epochs).item())

    print(f"End of itetration\nNumber of solutions found so far: {len(found_charges)}")

log_file.close()

fig, ax = plt.subplots(1, 1)
itierations = np.arange(1, num_iterations+1)
ax.plot(itierations, policy_losses, label="policy loss", color="r")
ax.plot(itierations, val_losses, label="val loss", color="b")
ax.plot(itierations, tot_losses, label="tot loss", color="m")

ax.legend(loc="upper right")
plt.show()
