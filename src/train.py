import numpy as np
from src.neural_net import *
from src.rwd_func import *
import matplotlib.pyplot as plt
import torch

# Choo Choo
# This is to avoid writing the same thing in main and hp_tuning

# I plan to eventually remove the agent's dependence on max_charge
def init_agent(num_epochs, num_transitions, minibatch_size, max_charge, actor_lr, critic_lr,
               actor_lr_gamma, critic_lr_gamma, actor_clip_epsilon, critic_clip_epsilon,
               gae_gamma, lmbda, entropy_coef):
    return PPO(num_epochs, num_transitions, minibatch_size, max_charge, actor_lr, critic_lr,
               actor_lr_gamma, critic_lr_gamma, actor_clip_epsilon, critic_clip_epsilon,
               gae_gamma, lmbda, entropy_coef)

def init_env(max_charge, max_steps, rwd):
    return Charge_Env(max_charge, max_steps, rwd)

def train(num_iterations, num_transitions, num_epochs, log_file, lr_upd_freq,
          agent, env, trial=None):
    pol_losses = []
    val_losses = []
    found_charges = []
    num_solutions = []

    for i in range(num_iterations):

        print(f"Currently on iteration {i+1} of {num_iterations}")

        states = []
        actions = []
        log_probs = []
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

        mean_rwd = env.rewards_sum / num_transitions
        print(f"Mean reward: {mean_rwd: .2f}")
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

            if num_epochs // (j+1) == lr_upd_freq:
                agent.actor_scheduler.step()
                agent.critic_scheduler.step()

            batch_pol_loss += batch_pol_loss_
            batch_val_loss += batch_val_loss_

        pol_losses.append((batch_pol_loss / num_epochs).detach())
        val_losses.append((batch_val_loss / num_epochs).detach())

        num_solutions.append(len(found_charges))
        print(f"End of iteration\nNumber of solutions found so far: {num_solutions[-1]}")

    if log_file is not None:
        log_file.close()

    return num_solutions, pol_losses, val_losses, mean_rwd if trial is not None else None

def plot(figname, num_iterations, num_solutions, pol_losses, val_losses):
    fig, axs = plt.subplots(1, 3)

    itierations = np.arange(1, num_iterations+1)
    axs[0].plot(itierations, pol_losses, color="r")
    axs[0].set_title("policy loss")

    axs[1].plot(itierations, val_losses, color="b")
    axs[1].set_title("value loss")
    
    axs[2].plot(itierations, num_solutions)
    axs[2].set_title("number of solutions")

    if figname is not None:
        fig.savefig(figname)
    else:
        plt.show()