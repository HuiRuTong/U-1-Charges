import torch
from src.charge_env import *

class Policy(torch.nn.Module):
    def __init__(self):
            super().__init__()
            self.encode = torch.nn.Sequential(
                torch.nn.Linear(3, 256),
            )
    
            self.choose_particle = torch.nn.Sequential(
                torch.nn.Linear(6*256, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 6)
            )
    
            self.choose_generation = torch.nn.Sequential(
                torch.nn.Linear(6*256, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 3)
            )
    
            self.choose_mod = torch.nn.Sequential(
                torch.nn.Linear(6*256, 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 2)
            )

    def forward(self, states):
        encoded = self.encode(states)
        reshaped = torch.reshape(encoded, (len(encoded), -1,))
        # An alternative would be to take the mean of the logits
        # over all 6 particles but that feels really wrong

        particle_logits = self.choose_particle(reshaped)
        generation_logits = self.choose_generation(reshaped)
        mod_logits = self.choose_mod(reshaped)

        return particle_logits, generation_logits, mod_logits

class Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(18, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )

    def forward(self, state):
        reshaped = torch.reshape(state, (len(state), -1,))
        val = self.critic(reshaped)

        return torch.flatten(val)

class PPO():
    def __init__(self, num_transitions, num_epochs, minibatch_size, max_charge, actor_lr, critic_lr,
                 actor_lr_gamma, critic_lr_gamma, actor_clip_epsilon, critic_clip_epsilon, gae_gamma,
                 lmbda, entropy_coef):
        self.num_transitions = num_transitions
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.max_charge = max_charge

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_lr_gamma = actor_lr_gamma
        self.critic_lr_gamma = critic_lr_gamma

        self.actor_clip_epsilon = actor_clip_epsilon
        self.critic_clip_epsilon = critic_clip_epsilon

        self.gae_gamma = gae_gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef

        self.actor = Policy()
        self.critic = Value()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), self.critic_lr)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, 20, self.actor_lr_gamma)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, 20, self.critic_lr_gamma)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.vals = []
        self.vals_offset = []
        self.rewards = []
        self.ended = []

        self.advantages = torch.zeros(self.num_transitions)
        self.vals_tar = torch.zeros(self.num_transitions)

    def get_action(self, states, get_entropy=False):
        particle_logits, generation_logits, mod_logits = self.actor.forward(states)
    
        particle_distr = torch.distributions.Categorical(logits=particle_logits)
        chosen_particle = particle_distr.sample()
        particle_log_prob = particle_distr.log_prob(chosen_particle)

        # For dependent 3rd gen charges
        generation_mask = torch.concat((torch.zeros((states.size()[0], 2), dtype=torch.bool),
                                        torch.unsqueeze(chosen_particle >= 2, dim=-1)), dim=-1)
        generation_logits.masked_fill_(generation_mask, -torch.inf)

        generation_distr = torch.distributions.Categorical(logits=generation_logits)
        chosen_generation = generation_distr.sample()
        generation_log_prob = generation_distr.log_prob(chosen_generation)

        # To prevent going oob
        # Note that the 3rd gen charges CAN STILL EXCEED the bounds
        mod_mask = torch.stack((states[torch.arange(states.size()[0]),
                                       chosen_particle, chosen_generation] <= -self.max_charge,
                                states[torch.arange(states.size()[0]),
                                       chosen_particle, chosen_generation] >= self.max_charge),
                                dim=-1)
        mod_logits.masked_fill_(mod_mask, -torch.inf)

        mod_distr = torch.distributions.Categorical(logits=mod_logits)
        chosen_mod = mod_distr.sample()
        mod_log_prob = mod_distr.log_prob(chosen_mod)

        if get_entropy:
            return (torch.stack((chosen_particle, chosen_generation, chosen_mod)),
                    torch.sum(torch.stack((particle_log_prob, generation_log_prob, mod_log_prob)), 0),
                    torch.mean(torch.stack((particle_distr.entropy(), generation_distr.entropy(), mod_distr.entropy()))))
        
        return (torch.tensor((chosen_particle.item(), chosen_generation.item(), chosen_mod.item())),
                torch.sum(torch.stack((particle_log_prob, generation_log_prob, mod_log_prob)), 0))

    def get_action_log_probs(self, states, actions, get_entropy=False):
        particle_logits, generation_logits, mod_logits = self.actor.forward(states)
        particle_distr = torch.distributions.Categorical(logits=particle_logits)
        chosen_particle = actions[:, 0]
        particle_log_prob = particle_distr.log_prob(chosen_particle)

        generation_mask = torch.concat((torch.zeros((states.size()[0], 2), dtype=torch.bool),
                                        torch.unsqueeze(chosen_particle >= 2, dim=-1)), dim=-1)
        generation_logits.masked_fill_(generation_mask, -torch.inf)

        generation_distr = torch.distributions.Categorical(logits=generation_logits)
        chosen_generation = actions[:, 1]
        generation_log_prob = generation_distr.log_prob(chosen_generation)

        mod_mask = torch.stack((states[torch.arange(states.size()[0]),
                                       chosen_particle, chosen_generation] <= -self.max_charge,
                                states[torch.arange(states.size()[0]),
                                       chosen_particle, chosen_generation] >= self.max_charge),
                                dim=-1)
        mod_logits.masked_fill(mod_mask, -torch.inf)

        mod_distr = torch.distributions.Categorical(logits=mod_logits)
        chosen_mod = actions[:, 2]
        mod_log_prob = mod_distr.log_prob(chosen_mod)

        if get_entropy:
            return (torch.sum(torch.stack((particle_log_prob, generation_log_prob, mod_log_prob)), 0),
                    torch.mean(torch.stack((particle_distr.entropy(), generation_distr.entropy(), mod_distr.entropy()))))
        return torch.sum(torch.stack((particle_log_prob, generation_log_prob, mod_log_prob)), 0)

    def calc_gae_tar(self):     # haha gay
        j = self.vals.size()[0] - 3  # index for values since its size depends on the number of terminal and truncated states

        self.advantages[-1] = self.rewards[-1] + self.gae_gamma*self.vals[-1] - self.vals[-2]
        self.vals_tar[-1] = torch.add(self.advantages[-1], self.vals[-2])
        for i in range(self.num_transitions-2, -1, -1):
            if self.ended[i]:
                j -= 1

            delta = self.rewards[i] + self.gae_gamma*self.vals[j+1] - self.vals[j]
            self.advantages[i] = self.gae_gamma*self.lmbda*(1-self.ended[i])*self.advantages[i+1] + delta
            self.vals_tar[i] = torch.add(self.advantages[i], self.vals[j])

            j -= 1
        self.advantages = (self.advantages - torch.mean(self.advantages)) / torch.std(self.advantages)

    def get_clip_obj(self, ratio, indices):
        obj = torch.tensor(0, dtype=torch.float32)

        j = 0   # index for ratio
        for i in indices:
            if (ratio[j] < 1 - self.actor_clip_epsilon) and (self.advantages[i] < 0):
                obj = obj + (1 - self.actor_clip_epsilon) * self.advantages[i]
            elif (ratio[j] > 1 + self.actor_clip_epsilon) and (self.advantages[i] > 0):
                obj = obj + (1 + self.actor_clip_epsilon) * self.advantages[i]
            else:
                obj = obj + ratio[j] * self.advantages[i]
            j += 1

        return obj / self.minibatch_size

    def get_clip_val(self, new_vals, indices):
        # This, along w/ the clipped loss are from OpenAI's PPO2
        clip_vals = torch.zeros((self.minibatch_size,))

        for j, i in enumerate(indices):
            # Add on the number of terminations / truncation
            # encountered up to the ith element in vals
            # this works because if S0, S1, S3, S4, S6 and V0, V1, V2, V3, V4, V5, V6
            # For vals to match with states and therefore new_vals,
            # its indices should be 0, 1, 3, 4, 6 as opposed to 0, 1, 2, 3, 4

            if new_vals[j] < self.vals[i+self.vals_offset[i]] - self.critic_clip_epsilon:
                clip_vals[j] = self.vals[i+self.vals_offset[i]] - self.critic_clip_epsilon
            elif new_vals[j] > self.vals[i+self.vals_offset[i]] + self.critic_clip_epsilon:
                clip_vals[j] = self.vals[i+self.vals_offset[i]] + self.critic_clip_epsilon
            else:
                clip_vals[j] = new_vals[j]

        return clip_vals

    def upd(self, indices):
        batch_pol_loss = 0
        batch_val_loss = 0
        num_batches = self.num_transitions / self.minibatch_size

        for j in range(self.num_transitions // self.minibatch_size):
            start = j * self.minibatch_size
            end = (j+1) * self.minibatch_size

            new_log_probs, entropy = self.get_action_log_probs(self.states[indices[start:end]],
                                                               self.actions[indices[start:end]],
                                                               get_entropy=True)
            ratio = torch.exp(torch.sub(new_log_probs, torch.flatten(self.log_probs[indices[start:end]])))
            obj = self.get_clip_obj(ratio, indices[start:end])
    
            pol_loss = -obj - self.entropy_coef*entropy

            new_vals = self.critic.forward(self.states[indices[start:end]])
            clip_vals = self.get_clip_val(new_vals, indices[start:end])
            val_loss = 0.5 * torch.mean(torch.maximum(torch.square(new_vals-self.vals_tar[indices[start:end]]),
                                                      torch.square(clip_vals-self.vals_tar[indices[start:end]])))
    
            self.actor_optimizer.zero_grad()
            pol_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            val_loss.backward()
            self.critic_optimizer.step()

            batch_pol_loss += pol_loss
            batch_val_loss += val_loss

        return batch_pol_loss / num_batches, batch_val_loss / num_batches