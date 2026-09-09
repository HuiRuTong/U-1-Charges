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

    def forward(self, state):
        encoded = self.encode(state)
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
    def __init__(self, num_transitions, num_epochs, minibatch_size, actor_lr, critic_lr,
                 actor_lr_gamma, critic_lr_gamma, actor_clip_epsilon, critic_clip_epsilon, gae_gamma,
                 lmbda, entropy_coef):
        self.num_transitions = num_transitions
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size

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
        self.rewards = []
        self.ended = []
        self.vals = []

        self.advantages = torch.zeros(self.num_transitions)
        self.vals_tar = torch.zeros(self.num_transitions)

    def calc_gae_tar(self):     # haha gay
        j = len(self.vals) - 3  # index for values since its size depends on the number of terminal and truncated states

        self.advantages[-1] = self.rewards[-1] + self.gae_gamma*self.vals[-1] - self.vals[-2]
        self.vals_tar[-1] = torch.add(self.advantages[-1], self.vals[-2])
        for i in range(self.num_transitions-2, -1, -1):
            if self.ended[i]:
                j -= 1

            delta = self.rewards[i] + self.gae_gamma*self.vals[j+1] - self.vals[i]
            self.advantages[i] = self.gae_gamma*self.lmbda*self.advantages[i+1] + delta
            self.vals_tar[i] = torch.add(self.advantages[i], self.vals[j])

            j -= 1
        self.advantages = (self.advantages - torch.mean(self.advantages)) / torch.std(self.advantages)

    def get_ratio(self, particle_distr, generation_distr, mod_distr, indices):
        particle_log_prob = particle_distr.log_prob(torch.flatten(self.actions[indices, 0]))
        generation_log_prob = generation_distr.log_prob(torch.flatten(self.actions[indices, 1]))
        mod_log_prob = mod_distr.log_prob(torch.flatten(self.actions[indices, 2]))

        new_log_probs = torch.sum(torch.stack((particle_log_prob, generation_log_prob, mod_log_prob)), 0)

        return torch.exp(torch.sub(new_log_probs, torch.flatten(self.log_probs[indices])))

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

        j = 0
        for i in indices:
            if new_vals[j] < self.vals[i] - self.critic_clip_epsilon:
                clip_vals[j] = self.vals[i] - self.critic_clip_epsilon
            elif new_vals[j] > self.vals[i] + self.critic_clip_epsilon:
                clip_vals[j] = self.vals[i] + self.critic_clip_epsilon
            else:
                clip_vals[j] = new_vals[j]

            j += 1

        return clip_vals

    def upd(self, indices):
        for j in range(self.num_transitions // self.minibatch_size):
            start = j * self.minibatch_size
            end = (j+1) * self.minibatch_size

            particle_logits, generation_logits, mod_logits = self.actor.forward(self.states[indices[start:end]])
            particle_distr = torch.distributions.Categorical(logits=particle_logits)
            generation_distr = torch.distributions.Categorical(logits=generation_logits)
            mod_distr = torch.distributions.Categorical(logits=mod_logits)

            ratio = self.get_ratio(particle_distr, generation_distr, mod_distr, indices[start:end])
            obj = self.get_clip_obj(ratio, indices[start:end])

            entropy = torch.mean(torch.stack(
                                (particle_distr.entropy(), generation_distr.entropy(), mod_distr.entropy())))
    
            pol_loss = -obj - self.entropy_coef*entropy

            new_vals = self.critic.forward(self.states[indices[start:end]])
            #print(new_vals.mean(), new_vals.std())


            clip_vals = self.get_clip_val(new_vals, indices[start:end])
            val_loss = 0.5 * torch.mean(torch.maximum(torch.square(new_vals-self.vals_tar[indices[start:end]]),
                                                      torch.square(clip_vals-self.vals_tar[indices[start:end]])))
    
            self.actor_optimizer.zero_grad()
            pol_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            val_loss.backward()
            self.critic_optimizer.step()

            """with torch.no_grad():
                check = self.critic(self.states[indices[start:end]])
                print("after: ", check.mean().item(), check.std().item())"""

            return pol_loss, val_loss