import torch
from src.charge_env import *

class ActorCritic(torch.nn.Module):
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

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(6*256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )

    def forward(self, state, action_only=False, value_only=False):
        encoded = self.encode(state)
        reshaped = torch.reshape(encoded, (len(encoded), -1,))
        # An alternative would be to take the mean of the logits
        # over all 6 particles but that feels really wrong

        if not action_only:
            val = self.critic(reshaped)

        if not value_only:
            particle_logits = self.choose_particle(reshaped)
            generation_logits = self.choose_generation(reshaped)
            mod_logits = self.choose_mod(reshaped)

        if action_only:
            return particle_logits, generation_logits, mod_logits
        elif value_only:
            return torch.flatten(val)
        else:
            return particle_logits, generation_logits, mod_logits, torch.flatten(val)

class PPO():
    def __init__(self, num_epochs=64, num_transitions=2048, minibatch_size=64, lr=0.5,
                 lr_gamma=0.1, gamma=0.98, lmbda=0.02, pol_clip_epsilon=0.2, val_clip_epsilon=0.2, entropy_coef=0.5):
        self.num_epochs = num_epochs
        self.num_transitions = num_transitions
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.gamma = gamma
        self.lmbda = lmbda
        self.pol_clip_epsilon = pol_clip_epsilon
        self.val_clip_epsilon = val_clip_epsilon
        self.entropy_coef = entropy_coef

        self.actor_critic = ActorCritic()
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 20, self.lr_gamma)

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

        self.advantages[-1] = self.rewards[-1] + self.gamma*self.vals[-1] - self.vals[-2]
        self.vals_tar[-1] = torch.add(self.advantages[-1], self.vals[-2])
        for i in range(self.num_transitions-2, -1, -1):
            if self.ended[i]:
                j -= 1

            delta = self.rewards[i] + self.gamma*self.vals[j+1] - self.vals[i]
            self.advantages[i] = self.gamma*self.lmbda*self.advantages[i+1] + delta
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
            if (ratio[j] < 1 - self.pol_clip_epsilon) and (self.advantages[i] < 0):
                obj = obj + (1 - self.pol_clip_epsilon) * self.advantages[i]
            elif (ratio[j] > 1 + self.pol_clip_epsilon) and (self.advantages[i] > 0):
                obj = obj + (1 + self.pol_clip_epsilon) * self.advantages[i]
            else:
                obj = obj + ratio[j] * self.advantages[i]
            j += 1

        return obj / self.minibatch_size

    def get_clip_val(self, new_vals, indices):
        # This, along w/ the clipped loss are from OpenAI's PPO2
        clip_vals = torch.zeros((self.minibatch_size,))

        j = 0
        for i in indices:
            if new_vals[j] < self.vals[i] - self.val_clip_epsilon:
                clip_vals[j] = self.vals[i] - self.val_clip_epsilon
            elif new_vals[j] > self.vals[i] + self.val_clip_epsilon:
                clip_vals[j] = self.vals[i] + self.val_clip_epsilon
            else:
                clip_vals[j] = new_vals[j]

            j += 1

        return clip_vals

    def upd(self, indices):
        for j in range(self.num_transitions // self.minibatch_size):
            start = j * self.minibatch_size
            end = (j+1) * self.minibatch_size

            particle_logits, generation_logits, mod_logits, new_vals = self.actor_critic.forward(self.states[indices[start:end]])        
            particle_distr = torch.distributions.Categorical(logits=particle_logits)
            generation_distr = torch.distributions.Categorical(logits=generation_logits)
            mod_distr = torch.distributions.Categorical(logits=mod_logits)

            ratio = self.get_ratio(particle_distr, generation_distr, mod_distr, indices[start:end])
            obj = self.get_clip_obj(ratio, indices[start:end])

            entropy = torch.mean(torch.stack(
                                (particle_distr.entropy(), generation_distr.entropy(), mod_distr.entropy())))
    
            policy_loss = -obj - self.entropy_coef*entropy

            clip_vals = self.get_clip_val(new_vals, indices[start:end])
            val_loss = 0.5 * torch.mean(torch.maximum(torch.square(new_vals-self.vals_tar[indices[start:end]]),
                                                      torch.square(clip_vals-self.vals_tar[indices[start:end]])))

            tot_loss = policy_loss + val_loss
    
            self.optimizer.zero_grad()
            tot_loss.backward()
            self.optimizer.step()

            return policy_loss, val_loss, tot_loss
