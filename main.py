import numpy as np
from src.neural_net import *
from src.rwd_func import *
import matplotlib.pyplot as plt
import torch

num_iterations = 256
num_transitions = 200
num_epochs = 20
minibatch_size = 20

actor_lr = 1e-5
critic_lr = 2.5e-6
actor_lr_gamma = 0.2
critic_lr_gamma = 0.1
actor_clip_epsilon = 0.2
critic_clip_epsilon = 0.2

lr_upd_freq = 5
gae_gamma = 0.85
lmbda = 0.95
entropy_coef = 0.02

max_charge = 5
max_steps = 25

agent = PPO(num_transitions, num_epochs, minibatch_size, actor_lr, critic_lr, actor_lr_gamma, critic_lr_gamma,
            actor_clip_epsilon, critic_clip_epsilon, gae_gamma, lmbda, entropy_coef)
env = Charge_Env(max_charge, max_steps, abs_err_rwd)

pol_losses = []
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

    for j in range(num_transitions):
        state = torch.tensor(env.charges, dtype=torch.float32)
        states.append(state)

        particle_logits, generation_logits, mod_logits = agent.actor(torch.unsqueeze(state, 0))
        
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

        actions.append(action)
        log_probs.append(log_prob)

        val = torch.flatten(agent.critic(torch.unsqueeze(state, 0)))
        vals.append(val)

        state, reward, terminated, truncated, info = env.step(action, found_charges, log_file)
        rwd_func.append(reward)

        ended.append(int(terminated or truncated))

        if terminated or truncated or j == num_transitions - 1:  # Last state also requires the next value
            state = torch.tensor(state, dtype=torch.float32)
            vals.append(int(not terminated) * agent.critic(torch.unsqueeze(state, 0)))   # Terminated states will hvae zero value

            env.reset()
    
    print(f"Mean rewards: {env.rewards_sum / num_transitions}")
    env.rewards_sum = 0.0

    agent.states = torch.stack(states).detach()
    agent.actions = torch.stack(actions).detach()
    agent.log_probs = torch.stack(log_probs).detach()
    agent.vals = torch.stack(vals).detach()
    agent.rewards = (torch.tensor(rwd_func))
    agent.ended = torch.tensor(ended)

    agent.rewards = ((agent.rewards - torch.mean(agent.rewards)) / torch.std(agent.rewards)).detach()
    agent.calc_gae_tar()

    pol_loss = 0
    val_loss = 0
    for j in range(num_epochs):
        print(f"\t Epoch {j+1} of {num_epochs}", end='')

        pol_loss_, val_loss_ = agent.upd(torch.randperm(num_transitions))
        pol_loss += pol_loss_
        val_loss += val_loss_
        print(f"\t policy loss: {pol_loss: .2f}, val loss: {val_loss: .2f}")

        """if num_epochs // (j+1) == lr_upd_freq:
            agent.actor_scheduler.step()
            agent.critic_scheduler.step()"""

    pol_losses.append((pol_loss / num_epochs).item())
    val_losses.append((val_loss / num_epochs).item())

    print(f"End of itetration\nNumber of solutions found so far: {len(found_charges)}")

log_file.close()

fig, axs = plt.subplots(1, 2)
itierations = np.arange(1, num_iterations+1)
axs[0].plot(itierations, pol_losses, label="policy loss", color="r")
axs[1].plot(itierations, val_losses, label="val loss", color="b")
plt.show()